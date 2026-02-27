from __future__ import annotations

from pathlib import Path
import argparse
import sys

from fba_llm.ingest import build_combined_facts_block
from fba_llm.advisor_text import run_advisor_text_strict
from fba_llm.guards import check_no_new_numbers, check_no_banned_claims

ROOT = Path(__file__).resolve().parent
INPUTS_DIR = ROOT / "inputs"
METRICS_DIR = INPUTS_DIR / "Metrics"
REVIEWS_DIR = INPUTS_DIR / "Reviews"


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
            f"Expected one of: {', '.join(exts)}"
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FBA_LLM CLI runner (API-based advisory output).")
    p.add_argument("--question", default="Analyze this for FBA viability.")
    p.add_argument("--metrics-file", default=None)
    p.add_argument("--reviews-file", default=None)
    p.add_argument("--no-metrics", action="store_true")
    p.add_argument("--no-reviews", action="store_true")
    p.add_argument("--facts-preview", type=int, default=800)
    p.add_argument("--deep-reviews", action="store_true", help="Extract themes/findings from review text (slower).")
    return p.parse_args()


def main() -> int:
    ensure_inputs_layout()
    args = parse_args()

    metrics_path = None
    reviews_path = None

    if not args.no_metrics:
        metrics_path = (
            Path(args.metrics_file).resolve()
            if args.metrics_file
            else pick_latest_file(METRICS_DIR, (".csv",), "Metrics CSV")
        )

    if not args.no_reviews:
        reviews_path = (
            Path(args.reviews_file).resolve()
            if args.reviews_file
            else pick_latest_file(REVIEWS_DIR, (".txt",), "Reviews TXT")
        )

    if metrics_path is None and reviews_path is None:
        eprint("No inputs enabled.")
        return 1

    facts_block = build_combined_facts_block(
        metrics_csv=metrics_path,
        reviews_txt=reviews_path,
        deep_review_analysis=args.deep_reviews,
    )

    if args.facts_preview > 0:
        eprint("\n--- FACTS BLOCK (preview) ---")
        eprint(facts_block[: args.facts_preview])
        eprint("--- END FACTS BLOCK ---\n")

    raw = run_advisor_text_strict(args.question, facts_block)

    ok_nums, extras = check_no_new_numbers(raw, facts_block)
    if not ok_nums:
        eprint("NUMBER GUARD TRIGGERED:")
        eprint(sorted(extras))
        return 2

    ok_claims, hits = check_no_banned_claims(raw, facts_block)
    if not ok_claims:
        eprint("CLAIM GUARD TRIGGERED:")
        eprint(hits)
        return 3

    print("\n=== FBA ADVISOR OUTPUT ===\n")
    print(raw.strip())
    print("\n=== END OUTPUT ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())