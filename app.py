from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

from fba_llm.model import find_latest_snapshot, load_model
from fba_llm.ingest import build_combined_facts_block
from fba_llm.advisor_text import run_advisor_text
from fba_llm.formatting import parse_advisor_text_to_json
from fba_llm.guards import check_no_new_numbers, check_no_banned_claims

ROOT = Path(__file__).resolve().parent

INPUTS_DIR = ROOT / "inputs"
METRICS_DIR = INPUTS_DIR / "Metrics"
REVIEWS_DIR = INPUTS_DIR / "Reviews"

DEFAULT_CACHE_ROOT = ROOT / "models" / "llama-2-7b-hf"


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def ensure_inputs_layout():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)


def pick_latest_file(folder: Path, exts: tuple[str, ...], label: str) -> Path:
    if not folder.exists():
        raise FileNotFoundError(f"{label} folder not found: {folder}")

    candidates = []
    for ext in exts:
        candidates.extend(folder.glob(f"*{ext}"))

    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No {label} file found in {folder}\n"
            f"Expected one of: {', '.join(exts)}\n"
            f"Fix: place a file in that folder."
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = candidates[0]

    if len(candidates) > 1:
        eprint(f"WARNING: Multiple {label} files found in {folder}. Using newest: {chosen.name}")
        eprint("Other candidates: " + ", ".join(p.name for p in candidates[1:5]) + ("..." if len(candidates) > 5 else ""))

    return chosen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FBA_LLM CLI runner (auto-detect Metrics/ + Reviews/ files).")

    p.add_argument("--question", default="Analyze this for FBA viability.", help="Question for the advisor")

    # Optional overrides: you can still specify an explicit file if you want
    p.add_argument("--metrics-file", default=None, help="Explicit metrics CSV path (overrides inputs/Metrics/*)")
    p.add_argument("--reviews-file", default=None, help="Explicit reviews TXT path (overrides inputs/Reviews/*)")

    p.add_argument("--no-metrics", action="store_true", help="Ignore metrics CSV")
    p.add_argument("--no-reviews", action="store_true", help="Ignore reviews TXT")

    p.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT), help="HF cache root (default: models/llama-2-7b-hf)")
    p.add_argument("--facts-preview", type=int, default=600, help="Preview N chars of facts to stderr (0 disables)")

    return p.parse_args()


def main() -> int:
    ensure_inputs_layout()
    args = parse_args()

    cache_root = Path(args.cache_root).expanduser().resolve()

    metrics_path = None
    reviews_path = None

    if not args.no_metrics:
        if args.metrics_file:
            metrics_path = Path(args.metrics_file).expanduser().resolve()
            if not metrics_path.exists():
                eprint(f"Metrics file not found: {metrics_path}")
                return 1
        else:
            metrics_path = pick_latest_file(METRICS_DIR, (".csv",), "Metrics CSV")

    if not args.no_reviews:
        if args.reviews_file:
            reviews_path = Path(args.reviews_file).expanduser().resolve()
            if not reviews_path.exists():
                eprint(f"Reviews file not found: {reviews_path}")
                return 1
        else:
            reviews_path = pick_latest_file(REVIEWS_DIR, (".txt",), "Reviews TXT")

    if metrics_path is None and reviews_path is None:
        eprint("No inputs enabled. Remove --no-metrics/--no-reviews.")
        return 1

    eprint(f"Metrics path: {metrics_path}" if metrics_path else "Metrics: (disabled)")
    eprint(f"Reviews path: {reviews_path}" if reviews_path else "Reviews: (disabled)")

    # Load model
    try:
        model_path = find_latest_snapshot(cache_root)
    except FileNotFoundError as e:
        eprint(str(e))
        eprint(f"Checked cache root: {cache_root}")
        return 1

    eprint(f"Using model snapshot: {model_path}")
    tokenizer, model = load_model(model_path)

    # Build combined facts
    facts_block = build_combined_facts_block(
        metrics_csv=metrics_path,
        reviews_txt=reviews_path,
        tokenizer=tokenizer,
        model=model,
    )

    if args.facts_preview and args.facts_preview > 0:
        eprint("\n--- FACTS BLOCK (preview) ---")
        eprint(facts_block[: args.facts_preview])
        eprint("--- END FACTS BLOCK ---\n")

    raw = run_advisor_text(tokenizer, model, args.question, facts_block)
    eprint("\n--- RAW MODEL OUTPUT (debug) ---\n")
    eprint(raw[:2000])
    eprint("\n--- END RAW ---\n")
    obj = parse_advisor_text_to_json(raw)

    json_text = json.dumps(obj, ensure_ascii=False)

    ok_nums, extras = check_no_new_numbers(json_text, facts_block)
    if not ok_nums:
        eprint("NUMBER GUARD TRIGGERED:")
        eprint(sorted(extras))
        eprint("\n--- RAW MODEL OUTPUT ---\n")
        eprint(raw)
        return 2

    ok_claims, hits = check_no_banned_claims(json_text, facts_block)
    if not ok_claims:
        eprint("CLAIM GUARD TRIGGERED:")
        eprint(hits)
        eprint("\n--- RAW MODEL OUTPUT ---\n")
        eprint(raw)
        return 3

    print(json.dumps(obj, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
