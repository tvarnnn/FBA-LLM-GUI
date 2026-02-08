from pathlib import Path
import csv
import statistics
from typing import Optional

from fba_llm.chunking import extract_text_findings, ChunkConfig


def read_txt(path: Path, max_chars: int = 6000) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[:max_chars]


def read_txt_full(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _to_float(x: str) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def summarize_csv(path: Path, max_rows: int = 2000) -> str:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        cols_lower = [c.lower() for c in cols]

        def col_pick(*candidates: str) -> Optional[str]:
            for cand in candidates:
                if cand in cols:
                    return cand
            for cand in candidates:
                if cand.lower() in cols_lower:
                    return cols[cols_lower.index(cand.lower())]
            return None

        col_price = col_pick("price", "avg_price", "price_usd")
        col_rating = col_pick("rating", "stars", "avg_rating")
        col_reviews = col_pick("review_count", "reviews", "reviews_count", "rating_count")
        col_weight = col_pick("weight_lbs", "weight", "shipping_weight", "weight_lb")
        col_category = col_pick("category", "cat")

        prices, ratings, reviews, weights = [], [], [], []
        categories = set()

        row_count = 0
        for row in reader:
            row_count += 1
            if row_count > max_rows:
                break

            if col_price:
                v = _to_float(row.get(col_price, ""))
                if v is not None:
                    prices.append(v)
            if col_rating:
                v = _to_float(row.get(col_rating, ""))
                if v is not None:
                    ratings.append(v)
            if col_reviews:
                v = _to_float(row.get(col_reviews, ""))
                if v is not None:
                    reviews.append(v)
            if col_weight:
                v = _to_float(row.get(col_weight, ""))
                if v is not None:
                    weights.append(v)
            if col_category:
                c = (row.get(col_category, "") or "").strip()
                if c:
                    categories.add(c)

    def stats_line(name: str, arr: list[float]) -> str:
        if not arr:
            return f"{name}: INSUFFICIENT DATA"
        mn, mx = min(arr), max(arr)
        avg = statistics.mean(arr)
        return f"{name}: count={len(arr)} min={mn:.2f} avg={avg:.2f} max={mx:.2f}"

    def safe_mean(arr: list[float]) -> Optional[float]:
        return statistics.mean(arr) if arr else None

    price_avg = safe_mean(prices)
    rating_avg = safe_mean(ratings)
    reviews_avg = safe_mean(reviews)
    weight_avg = safe_mean(weights)

    price_spread_pct = None
    if prices and price_avg and price_avg != 0:
        price_spread_pct = (max(prices) - min(prices)) / price_avg * 100.0

    rating_spread = None
    if ratings:
        rating_spread = max(ratings) - min(ratings)

    review_dominance = None
    if reviews and reviews_avg and reviews_avg != 0:
        review_dominance = max(reviews) / reviews_avg

    category_count = len(categories)

    def band_label(
        value: Optional[float],
        low: float,
        high: float,
        low_label: str,
        mid_label: str,
        high_label: str,
    ) -> str:
        if value is None:
            return "INSUFFICIENT DATA"
        if value < low:
            return low_label
        if value <= high:
            return mid_label
        return high_label

    price_competition = band_label(
        price_spread_pct,
        low=15.0,
        high=30.0,
        low_label="LOW (tight pricing band)",
        mid_label="MEDIUM",
        high_label="HIGH (price war risk)",
    )

    quality_consistency = band_label(
        rating_spread,
        low=0.30,
        high=0.60,
        low_label="STABLE",
        mid_label="MIXED",
        high_label="VOLATILE",
    )

    review_concentration = band_label(
        review_dominance,
        low=1.50,
        high=2.50,
        low_label="FRAGMENTED (no single dominant listing)",
        mid_label="MODERATE",
        high_label="DOMINANT (top listing likely hard to beat)",
    )

    shipping_complexity = band_label(
        weight_avg,
        low=1.50,
        high=3.00,
        low_label="LIGHT",
        mid_label="MEDIUM",
        high_label="HEAVY",
    )

    sample_confidence = "LOW (very small sample)" if row_count < 20 else "MEDIUM/HIGH (larger sample)"

    lines = [
        "FILE_TYPE: CSV",
        f"FILE_NAME: {path.name}",
        f"ROWS_READ: {row_count}",
        "COLUMNS: " + (", ".join(cols) if cols else "INSUFFICIENT DATA"),
        "",
        "COMPUTED_STATS (use ONLY these numbers):",
        stats_line("PRICE", prices),
        stats_line("RATING", ratings),
        stats_line("REVIEW_COUNT", reviews),
        stats_line("WEIGHT_LBS", weights),
        "",
        "DERIVED_NUMBERS (use ONLY these numbers):",
        f"PRICE_SPREAD_PCT: {price_spread_pct:.2f}" if price_spread_pct is not None else "PRICE_SPREAD_PCT: INSUFFICIENT DATA",
        f"RATING_SPREAD: {rating_spread:.2f}" if rating_spread is not None else "RATING_SPREAD: INSUFFICIENT DATA",
        f"REVIEW_DOMINANCE_RATIO: {review_dominance:.2f}" if review_dominance is not None else "REVIEW_DOMINANCE_RATIO: INSUFFICIENT DATA",
        f"CATEGORY_COUNT: {category_count}",
        "",
        "DERIVED_LABELS (no new numbers):",
        f"PRICE_COMPETITION: {price_competition}",
        f"QUALITY_CONSISTENCY: {quality_consistency}",
        f"REVIEW_CONCENTRATION: {review_concentration}",
        f"SHIPPING_COMPLEXITY: {shipping_complexity}",
        f"SAMPLE_CONFIDENCE: {sample_confidence}",
    ]

    if categories:
        sample = sorted(list(categories))[:10]
        lines.append(f"CATEGORIES_SAMPLE: {', '.join(sample)}")
    else:
        lines.append("CATEGORIES_SAMPLE: INSUFFICIENT DATA")

    lines += [
        "",
        "NOT_PRESENT_UNLESS_IN_COLUMNS:",
        "- BSR",
        "- sales volume / sales velocity",
        "- listing age / time on market",
        "- PPC / ad spend",
        "- fees breakdown",
        "- profit margin / COGS / landed cost",
        "- return rate",
    ]

    return "\n".join(lines)


def build_facts_block(
    file_path: Path,
    tokenizer=None,
    model=None,
    *,
    txt_preview_chars: int = 6000,
    txt_chunk_threshold_chars: int = 12000,
) -> str:
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return summarize_csv(file_path)

    if suffix == ".txt":
        full = read_txt_full(file_path)

        if len(full) <= txt_chunk_threshold_chars:
            preview = full[:txt_preview_chars]
            return (
                f"FILE_TYPE: TXT\n"
                f"FILE_NAME: {file_path.name}\n"
                f"CONTENT_PREVIEW:\n{preview}"
            )

        if tokenizer is None or model is None:
            raise ValueError(
                "Large .txt detected but tokenizer/model not provided to build_facts_block(). "
                "Pass tokenizer/model from app.py."
            )

        findings = extract_text_findings(tokenizer, model, full, ChunkConfig())
        return (
            f"FILE_TYPE: TXT\n"
            f"FILE_NAME: {file_path.name}\n"
            f"{findings}"
        )

    raise ValueError(f"Unsupported file type: {suffix}. Use .csv or .txt")

def build_combined_facts_block(
    *,
    metrics_csv: Optional[Path] = None,
    reviews_txt: Optional[Path] = None,
    tokenizer=None,
    model=None,
) -> str:
    sections: list[str] = []

    def add_section(title: str, content: str):
        sections.append(f"=== {title} ===\n{content}".strip())

    if metrics_csv is not None:
        if not metrics_csv.exists():
            raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")
        add_section("METRICS", summarize_csv(metrics_csv))

    if reviews_txt is not None:
        if not reviews_txt.exists():
            raise FileNotFoundError(f"Reviews TXT not found: {reviews_txt}")
        txt_block = build_facts_block(reviews_txt, tokenizer=tokenizer, model=model)
        add_section("REVIEWS", txt_block)

    if not sections:
        return "NO_INPUT_DATA: INSUFFICIENT DATA"

    return "\n\n".join(sections)
