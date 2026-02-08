from __future__ import annotations

import shutil
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from fba_llm.model import find_latest_snapshot, load_model
from fba_llm.ingest import build_combined_facts_block
from fba_llm.advisor_text import run_advisor_text
from fba_llm.guards import check_no_new_numbers, check_no_banned_claims

# Paths
ROOT = Path(__file__).resolve().parent
INPUTS_DIR = ROOT / "inputs"
METRICS_DIR = INPUTS_DIR / "Metrics"
REVIEWS_DIR = INPUTS_DIR / "Reviews"
DEFAULT_CACHE_ROOT = ROOT / "models" / "llama-2-7b-hf"

# Dark theme colors
DARK = {
    "bg": "#0f111a",
    "panel": "#151827",
    "fg": "#e6e6e6",
    "muted": "#a6accd",
    "border": "#2a2f45",
    "text_bg": "#0b0d14",
    "text_fg": "#e6e6e6",
    "select_bg": "#2b3a67",
}

LIGHT = {
    "bg": "#f5f5f5",
    "panel": "#ffffff",
    "fg": "#111111",
    "muted": "#444444",
    "border": "#cccccc",
    "text_bg": "#ffffff",
    "text_fg": "#111111",
    "select_bg": "#cfe3ff",
}


# Helpers
def ensure_layout():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)


def pick_latest_file(folder: Path, exts: tuple[str, ...]) -> Path | None:
    candidates: list[Path] = []
    for ext in exts:
        candidates.extend(folder.glob(f"*{ext}"))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def clear_folder(folder: Path, exts: tuple[str, ...]):
    for ext in exts:
        for p in folder.glob(f"*{ext}"):
            try:
                p.unlink()
            except Exception:
                pass


def copy_into_folder(src: Path, dst_folder: Path, *, clear_existing: bool) -> Path:
    dst_folder.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        clear_folder(dst_folder, (src.suffix.lower(),))
    dst = dst_folder / src.name
    shutil.copy2(src, dst)
    return dst


# GUI
class FbaGui(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("FBA_LLM – Local Advisor")
        self.geometry("1050x740")

        ensure_layout()

        # backend state
        self.tokenizer = None
        self.model = None
        self.model_loaded = False

        # UI state vars
        self.cache_root = tk.StringVar(value=str(DEFAULT_CACHE_ROOT))
        self.question_var = tk.StringVar(value="Analyze this for FBA viability.")
        self.clear_old_var = tk.BooleanVar(value=True)
        self.use_metrics_var = tk.BooleanVar(value=True)
        self.use_reviews_var = tk.BooleanVar(value=True)
        self.dark_mode = tk.BooleanVar(value=True)

        self.metrics_label_var = tk.StringVar(value="")
        self.reviews_label_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready. Click 'Load Model' first.")

        self.is_running = False

        self._build_ui()
        self._apply_theme()
        self._refresh_input_labels()

    #Theme
    def _apply_theme(self):
        palette = DARK if self.dark_mode.get() else LIGHT

        self.configure(bg=palette["bg"])
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".", background=palette["bg"], foreground=palette["fg"])
        style.configure("TFrame", background=palette["bg"])
        style.configure("TLabel", background=palette["bg"], foreground=palette["fg"])
        style.configure(
            "TLabelframe",
            background=palette["bg"],
            foreground=palette["fg"],
            bordercolor=palette["border"],
        )
        style.configure("TLabelframe.Label", background=palette["bg"], foreground=palette["fg"])

        style.configure(
            "TEntry",
            fieldbackground=palette["panel"],
            foreground=palette["fg"],
            bordercolor=palette["border"],
            insertcolor=palette["fg"],
        )
        style.configure("TButton", background=palette["panel"], foreground=palette["fg"], bordercolor=palette["border"])
        style.map("TButton", background=[("active", palette["panel"])], foreground=[("active", palette["fg"])])

        style.configure("TNotebook", background=palette["bg"], bordercolor=palette["border"])
        style.configure("TNotebook.Tab", background=palette["panel"], foreground=palette["fg"], padding=(10, 6))
        style.map("TNotebook.Tab", background=[("selected", palette["bg"])], foreground=[("selected", palette["fg"])])

        # tk.Text widgets (manual)
        for t in (getattr(self, "advisor_text", None), getattr(self, "facts_text", None)):
            if t is None:
                continue
            t.configure(
                bg=palette["text_bg"],
                fg=palette["text_fg"],
                insertbackground=palette["text_fg"],
                selectbackground=palette["select_bg"],
                selectforeground=palette["text_fg"],
                highlightthickness=1,
                highlightbackground=palette["border"],
                highlightcolor=palette["border"],
            )

    def _on_toggle_theme(self):
        self._apply_theme()

    # UI Builders
    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Cache root (folder containing models--*/snapshots/*):").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.cache_root, width=80).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(top, text="Load Model", command=self.on_load_model).grid(row=0, column=2, padx=6)

        ttk.Checkbutton(top, text="Dark mode", variable=self.dark_mode, command=self._on_toggle_theme).grid(
            row=0, column=3, sticky="e"
        )
        top.columnconfigure(1, weight=1)

        inputs = ttk.LabelFrame(self, text="Inputs", padding=10)
        inputs.pack(fill="x", padx=10, pady=8)

        ttk.Checkbutton(inputs, text="Use Metrics CSV", variable=self.use_metrics_var, command=self._refresh_input_labels)\
            .grid(row=0, column=0, sticky="w")
        ttk.Button(inputs, text="Upload Metrics CSV…", command=self.on_upload_metrics).grid(row=0, column=1, padx=6)
        ttk.Label(inputs, textvariable=self.metrics_label_var).grid(row=0, column=2, sticky="w")

        ttk.Checkbutton(inputs, text="Use Reviews TXT", variable=self.use_reviews_var, command=self._refresh_input_labels)\
            .grid(row=1, column=0, sticky="w")
        ttk.Button(inputs, text="Upload Reviews TXT…", command=self.on_upload_reviews).grid(row=1, column=1, padx=6)
        ttk.Label(inputs, textvariable=self.reviews_label_var).grid(row=1, column=2, sticky="w")

        ttk.Checkbutton(
            inputs,
            text="When uploading, clear old files in that folder",
            variable=self.clear_old_var
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))

        inputs.columnconfigure(2, weight=1)

        runbox = ttk.LabelFrame(self, text="Run", padding=10)
        runbox.pack(fill="x", padx=10, pady=8)

        ttk.Label(runbox, text="Question:").grid(row=0, column=0, sticky="w")
        ttk.Entry(runbox, textvariable=self.question_var, width=90).grid(row=0, column=1, sticky="we", padx=6)

        self.run_btn = ttk.Button(runbox, text="Run Analysis", command=self.on_run)
        self.run_btn.grid(row=0, column=2, padx=6)

        runbox.columnconfigure(1, weight=1)

        status = ttk.Frame(self, padding=(10, 0, 10, 10))
        status.pack(fill="x")
        ttk.Label(status, textvariable=self.status_var).pack(anchor="w")

        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True, padx=10, pady=10)

        self.advisor_text = tk.Text(tabs, wrap="word", font=("Consolas", 11))
        tabs.add(self.advisor_text, text="Advisor Output")

        self.facts_text = tk.Text(tabs, wrap="word", font=("Consolas", 10))
        tabs.add(self.facts_text, text="Facts Block (debug)")

    # Thread-safe UI methods
    def _ui_set_status(self, msg: str):
        self.after(0, lambda: self.status_var.set(msg))

    def _ui_fill_outputs(self, facts: str, raw: str):
        def do():
            self.facts_text.delete("1.0", "end")
            self.advisor_text.delete("1.0", "end")
            self.facts_text.insert("1.0", facts or "")
            self.advisor_text.insert("1.0", raw or "")
        self.after(0, do)

    def _ui_finish_run(self):
        def do():
            self.is_running = False
            self.run_btn.configure(state="normal")
        self.after(0, do)

    # Input labels
    def _refresh_input_labels(self):
        m = pick_latest_file(METRICS_DIR, (".csv",))
        r = pick_latest_file(REVIEWS_DIR, (".txt",))

        mtxt = f"Current: {m.name}" if m else "Current: (none)"
        rtxt = f"Current: {r.name}" if r else "Current: (none)"

        if not self.use_metrics_var.get():
            mtxt += "  [disabled]"
        if not self.use_reviews_var.get():
            rtxt += "  [disabled]"

        self.metrics_label_var.set(mtxt)
        self.reviews_label_var.set(rtxt)

    # Upload handlers
    def on_upload_metrics(self):
        path = filedialog.askopenfilename(
            title="Select Metrics CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            dst = copy_into_folder(Path(path), METRICS_DIR, clear_existing=self.clear_old_var.get())
            self._refresh_input_labels()
            self._ui_set_status(f"Uploaded metrics: {dst.name}")
        except Exception as e:
            messagebox.showerror("Upload error", str(e))

    def on_upload_reviews(self):
        path = filedialog.askopenfilename(
            title="Select Reviews TXT",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            dst = copy_into_folder(Path(path), REVIEWS_DIR, clear_existing=self.clear_old_var.get())
            self._refresh_input_labels()
            self._ui_set_status(f"Uploaded reviews: {dst.name}")
        except Exception as e:
            messagebox.showerror("Upload error", str(e))

    # Model loading
    def on_load_model(self):
        if self.model_loaded:
            self._ui_set_status("Model already loaded.")
            return

        def load_worker():
            try:
                self._ui_set_status("Loading model (one-time)…")
                cache_root = Path(self.cache_root.get()).expanduser().resolve()
                model_path = find_latest_snapshot(cache_root)
                tokenizer, model = load_model(model_path)
                self.tokenizer = tokenizer
                self.model = model
                self.model_loaded = True
                self._ui_set_status(f"Model loaded ✅ ({model_path.name})")
            except Exception:
                err_msg = traceback.format_exc()
                self._ui_set_status("Model load failed ❌")
                self.after(0, messagebox.showerror, "Model load failed", err_msg)

        threading.Thread(target=load_worker, daemon=True).start()

    # Run pipeline
    def on_run(self):
        if self.is_running:
            return
        if not self.model_loaded:
            messagebox.showinfo("Model not loaded", "Click 'Load Model' first.")
            return

        use_metrics = self.use_metrics_var.get()
        use_reviews = self.use_reviews_var.get()
        if not use_metrics and not use_reviews:
            messagebox.showinfo("No inputs", "Enable Metrics and/or Reviews.")
            return

        metrics_path = pick_latest_file(METRICS_DIR, (".csv",)) if use_metrics else None
        reviews_path = pick_latest_file(REVIEWS_DIR, (".txt",)) if use_reviews else None

        if use_metrics and metrics_path is None:
            messagebox.showinfo("Missing Metrics", "No .csv found in inputs/Metrics/")
            return
        if use_reviews and reviews_path is None:
            messagebox.showinfo("Missing Reviews", "No .txt found in inputs/Reviews/")
            return

        question = (self.question_var.get() or "").strip() or "Analyze this for FBA viability."

        self.is_running = True
        self.run_btn.configure(state="disabled")

        # clear outputs
        self.advisor_text.delete("1.0", "end")
        self.facts_text.delete("1.0", "end")

        def run_worker():
            raw = ""
            facts_block = ""
            try:
                self._ui_set_status("Building facts block…")

                facts_block = build_combined_facts_block(
                    metrics_csv=metrics_path,
                    reviews_txt=reviews_path,
                    tokenizer=self.tokenizer,
                    model=self.model,
                )

                self._ui_set_status("Running model… (can take a bit)")

                raw = run_advisor_text(self.tokenizer, self.model, question, facts_block)

                # Guards on RAW
                ok_nums_raw, extras_raw = check_no_new_numbers(raw, facts_block)
                if not ok_nums_raw:
                    raise ValueError(f"NUMBER GUARD TRIGGERED (raw): {sorted(extras_raw)}")

                ok_claims_raw, hits_raw = check_no_banned_claims(raw, facts_block)
                if not ok_claims_raw:
                    raise ValueError(f"CLAIM GUARD TRIGGERED (raw): {hits_raw}")

                self._ui_fill_outputs(facts_block, raw)
                self._ui_set_status("Done ✅")

            except Exception as e:
                err_msg = traceback.format_exc()
                self._ui_set_status("Run failed ❌")

                e_msg = str(e)
                raw_preview = (raw or "")[:1500]

                # Still show what we got, so you can debug without rerunning.
                self._ui_fill_outputs(facts_block or "", raw or "")

                self.after(
                    0,
                    messagebox.showerror,
                    "Run failed",
                    f"{e_msg}\n\nDetails:\n{err_msg}\n\nRaw (first 1500 chars):\n{raw_preview}",
                )
            finally:
                self._ui_finish_run()

        threading.Thread(target=run_worker, daemon=True).start()


if __name__ == "__main__":
    app = FbaGui()
    app.mainloop()
