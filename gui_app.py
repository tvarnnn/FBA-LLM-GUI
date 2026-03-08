from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (same folder as this gui_app.py)
load_dotenv(Path(__file__).resolve().parent / ".env")

import os
import shutil
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from fba_llm.analysis import Assumptions
from fba_llm.analysis_session import AnalysisSession
from fba_llm.llm_backend import generate_text

# Paths
ROOT = Path(__file__).resolve().parent
INPUTS_DIR = ROOT / "inputs"
METRICS_DIR = INPUTS_DIR / "Metrics"
REVIEWS_DIR = INPUTS_DIR / "Reviews"
IMAGES_DIR = INPUTS_DIR / "Images"

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


def ensure_layout():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


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


class FbaGui(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("FBA_LLM – Research Copilot")
        self.geometry("1180x940")

        ensure_layout()

        # Session state
        self.session: AnalysisSession | None = None

        # UI state vars
        self.question_var = tk.StringVar(value="Give a cautious screening summary based only on the provided data.")
        self.followup_var = tk.StringVar(value="")
        self.clear_old_var = tk.BooleanVar(value=True)

        self.use_metrics_var = tk.BooleanVar(value=True)
        self.use_reviews_var = tk.BooleanVar(value=True)
        self.use_png_var = tk.BooleanVar(value=True)

        self.deep_reviews_var = tk.BooleanVar(value=True)
        self.dark_mode = tk.BooleanVar(value=True)

        # Provider selection
        self.provider_var = tk.StringVar(value=(os.getenv("LLM_PROVIDER") or "groq").strip().lower())
        self.model_override_var = tk.StringVar(value="")  # optional override

        # Assumptions controls
        self.lead_time_days_var = tk.StringVar(value="60")
        self.target_margin_pct_var = tk.StringVar(value="30")
        self.ad_spend_per_unit_var = tk.StringVar(value="0")

        self.metrics_label_var = tk.StringVar(value="")
        self.reviews_label_var = tk.StringVar(value="")
        self.png_label_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready.")

        # Progress UI vars
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_label_var = tk.StringVar(value="")

        self.is_running = False
        self._cancel_event = threading.Event()

        self._build_ui()
        self._apply_theme()
        self._refresh_input_labels()
        self._ui_progress_reset()

    # Theme
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

        style.configure("TProgressbar", troughcolor=palette["panel"])

        for t in (
            getattr(self, "advisor_text", None),
            getattr(self, "facts_text", None),
        ):
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

    # Provider wiring
    def _apply_provider_env(self):
        provider = (self.provider_var.get() or "groq").strip().lower()
        os.environ["LLM_PROVIDER"] = provider

        override = (self.model_override_var.get() or "").strip()
        if provider == "groq":
            if override:
                os.environ["GROQ_MODEL"] = override
        elif provider in ("anthropic", "claude"):
            if override:
                os.environ["ANTHROPIC_MODEL"] = override

    def _parse_assumptions(self) -> Assumptions:
        def to_int(s: str, default: int) -> int:
            try:
                return int(str(s).strip())
            except Exception:
                return default

        def to_float(s: str, default: float) -> float:
            try:
                return float(str(s).strip())
            except Exception:
                return default

        lead = to_int(self.lead_time_days_var.get(), 60)
        lead = max(1, min(lead, 180))

        margin = to_float(self.target_margin_pct_var.get(), 30.0)
        margin = max(-50.0, min(margin, 90.0))

        ad = to_float(self.ad_spend_per_unit_var.get(), 0.0)
        ad = max(0.0, min(ad, 9999.0))

        return Assumptions(
            lead_time_days=lead,
            target_margin_pct=margin,
            ad_spend_per_unit=ad,
        )

    # UI Builders
    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Provider:").grid(row=0, column=0, sticky="w")

        provider_box = ttk.Combobox(
            top,
            textvariable=self.provider_var,
            values=["groq", "anthropic"],
            width=14,
            state="readonly",
        )
        provider_box.grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(top, text="Model override (optional):").grid(row=0, column=2, sticky="w", padx=(18, 0))
        ttk.Entry(top, textvariable=self.model_override_var, width=40).grid(row=0, column=3, sticky="we", padx=6)

        ttk.Button(top, text="Test API", command=self.on_test_api).grid(row=0, column=4, padx=6)

        ttk.Checkbutton(top, text="Dark mode", variable=self.dark_mode, command=self._on_toggle_theme).grid(
            row=0, column=5, sticky="e"
        )

        top.columnconfigure(3, weight=1)

        assumptions = ttk.LabelFrame(self, text="Assumptions (used for computed analysis)", padding=10)
        assumptions.pack(fill="x", padx=10, pady=(0, 8))

        ttk.Label(assumptions, text="Lead time (days):").grid(row=0, column=0, sticky="w")
        ttk.Entry(assumptions, textvariable=self.lead_time_days_var, width=10).grid(row=0, column=1, sticky="w", padx=(6, 18))

        ttk.Label(assumptions, text="Target margin (%):").grid(row=0, column=2, sticky="w")
        ttk.Entry(assumptions, textvariable=self.target_margin_pct_var, width=10).grid(row=0, column=3, sticky="w", padx=(6, 18))

        ttk.Label(assumptions, text="Ad spend / unit ($):").grid(row=0, column=4, sticky="w")
        ttk.Entry(assumptions, textvariable=self.ad_spend_per_unit_var, width=10).grid(row=0, column=5, sticky="w", padx=(6, 0))

        assumptions.columnconfigure(6, weight=1)

        inputs = ttk.LabelFrame(self, text="Inputs", padding=10)
        inputs.pack(fill="x", padx=10, pady=8)

        ttk.Checkbutton(inputs, text="Use Metrics CSV", variable=self.use_metrics_var, command=self._refresh_input_labels).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(inputs, text="Upload Metrics CSV…", command=self.on_upload_metrics).grid(row=0, column=1, padx=6)
        ttk.Label(inputs, textvariable=self.metrics_label_var).grid(row=0, column=2, sticky="w")

        ttk.Checkbutton(inputs, text="Use Reviews TXT", variable=self.use_reviews_var, command=self._refresh_input_labels).grid(
            row=1, column=0, sticky="w"
        )
        ttk.Button(inputs, text="Upload Reviews TXT…", command=self.on_upload_reviews).grid(row=1, column=1, padx=6)
        ttk.Label(inputs, textvariable=self.reviews_label_var).grid(row=1, column=2, sticky="w")

        ttk.Checkbutton(inputs, text="Use PNG Evidence", variable=self.use_png_var, command=self._refresh_input_labels).grid(
            row=2, column=0, sticky="w"
        )
        ttk.Button(inputs, text="Upload PNG…", command=self.on_upload_png).grid(row=2, column=1, padx=6)
        ttk.Label(inputs, textvariable=self.png_label_var).grid(row=2, column=2, sticky="w")

        ttk.Checkbutton(
            inputs,
            text="Deep review analysis (slower, extracts themes/findings)",
            variable=self.deep_reviews_var,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))

        ttk.Checkbutton(
            inputs,
            text="When uploading, clear old files in that folder",
            variable=self.clear_old_var,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))

        inputs.columnconfigure(2, weight=1)

        runbox = ttk.LabelFrame(self, text="Build Screening Summary", padding=10)
        runbox.pack(fill="x", padx=10, pady=8)

        ttk.Label(runbox, text="Goal:").grid(row=0, column=0, sticky="w")
        ttk.Entry(runbox, textvariable=self.question_var, width=90).grid(row=0, column=1, sticky="we", padx=6)

        self.run_btn = ttk.Button(runbox, text="Build Analysis", command=self.on_run)
        self.run_btn.grid(row=0, column=2, padx=6)

        self.cancel_btn = ttk.Button(runbox, text="Cancel", command=self.on_cancel, state="disabled")
        self.cancel_btn.grid(row=0, column=3, padx=6)

        runbox.columnconfigure(1, weight=1)

        followup = ttk.LabelFrame(self, text="Ask Follow-up Question", padding=10)
        followup.pack(fill="x", padx=10, pady=8)

        ttk.Label(followup, text="Question:").grid(row=0, column=0, sticky="w")
        ttk.Entry(followup, textvariable=self.followup_var, width=90).grid(row=0, column=1, sticky="we", padx=6)

        self.ask_btn = ttk.Button(followup, text="Ask", command=self.on_ask_followup)
        self.ask_btn.grid(row=0, column=2, padx=6)
        followup.columnconfigure(1, weight=1)

        status = ttk.Frame(self, padding=(10, 0, 10, 6))
        status.pack(fill="x")
        ttk.Label(status, textvariable=self.status_var).pack(anchor="w")

        prog = ttk.Frame(self, padding=(10, 0, 10, 10))
        prog.pack(fill="x")

        ttk.Label(prog, textvariable=self.progress_label_var).pack(anchor="w")
        self.progress = ttk.Progressbar(
            prog,
            variable=self.progress_var,
            maximum=100.0,
            mode="determinate",
        )
        self.progress.pack(fill="x", pady=(4, 0))

        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True, padx=10, pady=10)

        self.advisor_text = tk.Text(tabs, wrap="word", font=("Consolas", 11))
        tabs.add(self.advisor_text, text="Analysis Output")

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

    def _ui_append_followup(self, question: str, answer: str):
        def do():
            sep = "\n\n" + "─" * 60 + "\n"
            self.advisor_text.insert("end", sep)
            self.advisor_text.insert("end", f"Q: {question}\n\n")
            self.advisor_text.insert("end", answer)
            self.advisor_text.see("end")

        self.after(0, do)

    def _ui_progress(self, label: str = "", value: float | None = None, *, indeterminate: bool = False):
        def do():
            self.progress_label_var.set(label or "")

            if indeterminate:
                if str(self.progress["mode"]) != "indeterminate":
                    self.progress.configure(mode="indeterminate")
                self.progress.start(12)
            else:
                try:
                    self.progress.stop()
                except Exception:
                    pass
                if str(self.progress["mode"]) != "determinate":
                    self.progress.configure(mode="determinate")
                if value is not None:
                    self.progress_var.set(float(value))

        self.after(0, do)

    def _ui_progress_reset(self):
        self._ui_progress("", 0.0, indeterminate=False)

    def _ui_progress_done(self):
        self._ui_progress("Complete", 100.0, indeterminate=False)

    def _ui_finish_run(self):
        def do():
            self.is_running = False
            self.run_btn.configure(state="normal")
            self.ask_btn.configure(state="normal")
            self.cancel_btn.configure(state="disabled")
            try:
                self.progress.stop()
            except Exception:
                pass

        self.after(0, do)

    def on_cancel(self):
        if self.is_running:
            self._cancel_event.set()
            self._ui_set_status("Cancelling... (waiting for API response to return)")
            self.cancel_btn.configure(state="disabled")

    # Input labels
    def _refresh_input_labels(self):
        m = pick_latest_file(METRICS_DIR, (".csv",))
        r = pick_latest_file(REVIEWS_DIR, (".txt",))
        p = pick_latest_file(IMAGES_DIR, (".png",))

        mtxt = f"Current: {m.name}" if m else "Current: (none)"
        rtxt = f"Current: {r.name}" if r else "Current: (none)"
        ptxt = f"Current: {p.name}" if p else "Current: (none)"

        if not self.use_metrics_var.get():
            mtxt += "  [disabled]"
        if not self.use_reviews_var.get():
            rtxt += "  [disabled]"
        if not self.use_png_var.get():
            ptxt += "  [disabled]"

        self.metrics_label_var.set(mtxt)
        self.reviews_label_var.set(rtxt)
        self.png_label_var.set(ptxt)

    # Upload handlers
    def _upload_file(self, title: str, filetypes: list, dst_dir: Path, label: str):
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if not path:
            return
        try:
            dst = copy_into_folder(Path(path), dst_dir, clear_existing=self.clear_old_var.get())
            self._refresh_input_labels()
            self._ui_set_status(f"Uploaded {label}: {dst.name}")
        except Exception as e:
            messagebox.showerror("Upload error", str(e))

    def on_upload_metrics(self):
        self._upload_file("Select Metrics CSV", [("CSV files", "*.csv"), ("All files", "*.*")], METRICS_DIR, "metrics")

    def on_upload_reviews(self):
        self._upload_file("Select Reviews TXT", [("Text files", "*.txt"), ("All files", "*.*")], REVIEWS_DIR, "reviews")

    def on_upload_png(self):
        self._upload_file("Select PNG", [("PNG files", "*.png"), ("All files", "*.*")], IMAGES_DIR, "PNG")

    # API test
    def on_test_api(self):
        self._apply_provider_env()

        def worker():
            try:
                self._ui_set_status("Testing API…")
                self._ui_progress("Testing API…", 10.0, indeterminate=True)

                out = generate_text("Reply with exactly: OK", max_tokens=8, temperature=0.0)
                if "OK" not in (out or ""):
                    raise ValueError(f"Unexpected response: {out!r}")

                self._ui_set_status("API OK")
                self._ui_progress("API OK", 100.0, indeterminate=False)
            except Exception:
                err = traceback.format_exc()
                self._ui_set_status("API test failed")
                self._ui_progress("API test failed", 0.0, indeterminate=False)
                self.after(0, messagebox.showerror, "API test failed", err)

        threading.Thread(target=worker, daemon=True).start()

    # Build screening summary
    def on_run(self):
        if self.is_running:
            return

        self._apply_provider_env()

        use_metrics = self.use_metrics_var.get()
        use_reviews = self.use_reviews_var.get()
        use_png = self.use_png_var.get()

        if not use_metrics and not use_reviews and not use_png:
            messagebox.showinfo("No inputs", "Enable Metrics, Reviews, and/or PNG.")
            return

        metrics_path = pick_latest_file(METRICS_DIR, (".csv",)) if use_metrics else None
        reviews_path = pick_latest_file(REVIEWS_DIR, (".txt",)) if use_reviews else None
        png_path = pick_latest_file(IMAGES_DIR, (".png",)) if use_png else None

        if use_metrics and metrics_path is None:
            messagebox.showinfo("Missing Metrics", "No .csv found in inputs/Metrics/")
            return
        if use_reviews and reviews_path is None:
            messagebox.showinfo("Missing Reviews", "No .txt found in inputs/Reviews/")
            return
        if use_png and png_path is None:
            messagebox.showinfo("Missing PNG", "No .png found in inputs/Images/")
            return

        question = (self.question_var.get() or "").strip() or "Give a cautious screening summary based only on the provided data."
        deep_reviews = bool(self.deep_reviews_var.get())
        assumptions = self._parse_assumptions()

        self.is_running = True
        self._cancel_event.clear()
        self.run_btn.configure(state="disabled")
        self.ask_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")

        self.advisor_text.delete("1.0", "end")
        self.facts_text.delete("1.0", "end")
        self._ui_progress_reset()
        self._ui_progress("Queued…", 0.0)

        def run_worker():
            try:
                self._ui_progress("Building analysis session…", 20.0)
                self._ui_set_status("Building analysis session…")

                session = AnalysisSession(
                    question=question,
                    assumptions=assumptions,
                    metrics_csv=metrics_path,
                    reviews_txt=reviews_path,
                    png_path=png_path,
                    deep_review_analysis=deep_reviews,
                )

                session.build()

                if self._cancel_event.is_set():
                    self._ui_set_status("Cancelled.")
                    self._ui_progress("Cancelled", 0.0, indeterminate=False)
                    return

                self._ui_progress("Generating screening summary…", 70.0, indeterminate=True)
                self._ui_set_status("Generating screening summary…")

                summary = session.generate_screening_summary()

                if self._cancel_event.is_set():
                    self._ui_set_status("Cancelled.")
                    self._ui_progress("Cancelled", 0.0, indeterminate=False)
                    return

                self.session = session

                self._ui_progress("Rendering output…", 95.0, indeterminate=False)
                self._ui_fill_outputs(session.facts_block, summary)
                self._ui_set_status("Done")
                self._ui_progress_done()

            except Exception as e:
                if self._cancel_event.is_set():
                    self._ui_set_status("Cancelled.")
                    self._ui_progress("Cancelled", 0.0, indeterminate=False)
                    return
                err_msg = traceback.format_exc()
                self._ui_set_status("Run failed")
                self._ui_progress("Run failed", 0.0, indeterminate=False)

                self.after(
                    0,
                    messagebox.showerror,
                    "Run failed",
                    f"{str(e)}\n\nDetails:\n{err_msg}",
                )
            finally:
                self._ui_finish_run()

        threading.Thread(target=run_worker, daemon=True).start()

    # Ask follow-up question
    def on_ask_followup(self):
        if self.is_running:
            return

        if self.session is None:
            messagebox.showinfo("No session", "Build an analysis first.")
            return

        question = (self.followup_var.get() or "").strip()
        if not question:
            messagebox.showinfo("Missing question", "Enter a follow-up question.")
            return

        self._apply_provider_env()

        self.is_running = True
        self._cancel_event.clear()
        self.run_btn.configure(state="disabled")
        self.ask_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")

        self._ui_progress_reset()
        self._ui_progress("Queued…", 0.0)

        def worker():
            try:
                self._ui_set_status("Answering follow-up question…")
                self._ui_progress("Answering follow-up question…", 60.0, indeterminate=True)

                answer = self.session.ask(question)

                if self._cancel_event.is_set():
                    self._ui_set_status("Cancelled.")
                    self._ui_progress("Cancelled", 0.0, indeterminate=False)
                    return

                self._ui_append_followup(question, answer)
                self._ui_set_status("Done")
                self._ui_progress_done()
            except Exception:
                if self._cancel_event.is_set():
                    self._ui_set_status("Cancelled.")
                    self._ui_progress("Cancelled", 0.0, indeterminate=False)
                    return
                err = traceback.format_exc()
                self._ui_set_status("Follow-up failed")
                self._ui_progress("Follow-up failed", 0.0, indeterminate=False)
                self.after(0, messagebox.showerror, "Follow-up failed", err)
            finally:
                self._ui_finish_run()

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    app = FbaGui()
    app.mainloop()