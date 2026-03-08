from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path
from typing import Optional, List

from fba_llm.llm_backend import generate_text
from fba_llm.chunking import ChunkConfig


def _dedupe_lines(text: str, *, max_unique_lines: int = 4000) -> str:
    seen = set()
    out_lines = []
    for line in (text or "").splitlines():
        if line in seen:
            continue
        seen.add(line)
        out_lines.append(line)
        if len(out_lines) >= max_unique_lines:
            break
    return "\n".join(out_lines)


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


# Review theme block helpers
def _looks_valid_theme_block(t: str) -> bool:
    t = (t or "").strip()
    if "TEXT_PRAISE:" not in t or "TEXT_COMPLAINTS:" not in t:
        return False
    bullets = [ln for ln in t.splitlines() if ln.strip().startswith("-")]
    return len(bullets) >= 6


def _sanitize_theme_block(t: str) -> str:
    t = (t or "").strip()
    if not t:
        return ""

    t = re.sub(r"(?im)^\s*text\s*prai?ze\s*:\s*$", "TEXT_PRAISE:", t)
    t = re.sub(r"(?im)^\s*text\s*praise\s*:\s*$", "TEXT_PRAISE:", t)
    t = re.sub(r"(?im)^\s*text\s*complaints?\s*:\s*$", "TEXT_COMPLAINTS:", t)

    if "TEXT_PRAISE:" not in t or "TEXT_COMPLAINTS:" not in t:
        return ""

    lines = [ln.rstrip() for ln in t.splitlines()]

    kept: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s in ("TEXT_PRAISE:", "TEXT_COMPLAINTS:"):
            kept.append(s)
            continue
        if s.startswith("-"):
            kept.append("- " + s[1:].strip())
            continue

    t2 = "\n".join(kept).strip()

    praise: List[str] = []
    complaints: List[str] = []
    section = None
    for ln in t2.splitlines():
        if ln == "TEXT_PRAISE:":
            section = "p"
            continue
        if ln == "TEXT_COMPLAINTS:":
            section = "c"
            continue
        if ln.startswith("-"):
            item = ln[1:].strip()
            if len(item) < 3:
                continue
            if section == "p":
                praise.append(item)
            elif section == "c":
                complaints.append(item)

    if len(praise) < 3 or len(complaints) < 3:
        return ""

    return (
        "TEXT_PRAISE:\n"
        f"- {praise[0]}\n"
        f"- {praise[1]}\n"
        f"- {praise[2]}\n"
        "TEXT_COMPLAINTS:\n"
        f"- {complaints[0]}\n"
        f"- {complaints[1]}\n"
        f"- {complaints[2]}"
    )


def summarize_reviews_themes(full_text: str, *, max_chars: int = 24000) -> str:
    src = (full_text or "")[:max_chars]

    prompt = (
        "You are extracting review themes for an Amazon product.\n"
        "Output EXACTLY this format (no extra lines):\n"
        "TEXT_PRAISE:\n"
        "- <theme>\n"
        "- <theme>\n"
        "- <theme>\n"
        "TEXT_COMPLAINTS:\n"
        "- <theme>\n"
        "- <theme>\n"
        "- <theme>\n\n"
        "REVIEWS TEXT:\n"
        f"{src}\n\n"
        "ANSWER:\n"
    )

    text = (generate_text(prompt, max_tokens=260, temperature=0.0, timeout_s=90) or "").strip()

    sanitized = _sanitize_theme_block(text)
    if sanitized and _looks_valid_theme_block(sanitized):
        return sanitized
    if _looks_valid_theme_block(text):
        return text

    return (
        "TEXT_PRAISE:\n"
        "- INSUFFICIENT DATA\n"
        "- INSUFFICIENT DATA\n"
        "- INSUFFICIENT DATA\n"
        "TEXT_COMPLAINTS:\n"
        "- INSUFFICIENT DATA\n"
        "- INSUFFICIENT DATA\n"
        "- INSUFFICIENT DATA"
    )


def extract_text_findings_api(text: str, cfg: ChunkConfig = ChunkConfig()) -> str:
    src = (text or "")[: (cfg.chunk_size * cfg.max_chunks)]
    if not src.strip():
        return "TEXT_FINDINGS:\n- INSUFFICIENT DATA"

    prompt = (
        "Extract actionable evidence from customer reviews.\n"
        "Rules:\n"
        "- Use ONLY the provided REVIEWS TEXT.\n"
        "- Hyphen bullets only.\n"
        f"- At most {cfg.max_total_bullets} bullets.\n\n"
        "Output format:\n"
        "TEXT_FINDINGS:\n"
        "- <bullet>\n\n"
        "REVIEWS TEXT:\n"
        f"{src}\n\n"
        "ANSWER:\n"
    )

    out = (generate_text(prompt, max_tokens=320, temperature=0.0, timeout_s=90) or "").strip()

    bullets: List[str] = []
    for ln in out.splitlines():
        s = ln.strip()
        if not s or s.lower().startswith("text_findings"):
            continue
        if s.startswith("-"):
            item = s[1:].strip()
        else:
            item = re.sub(r"^\s*\d+\s*[\.\)\-:]\s*", "", s).strip()
        item = item.strip(" -•\t")
        if len(item) >= 8:
            bullets.append(item)

    if not bullets:
        return "TEXT_FINDINGS:\n- INSUFFICIENT DATA"

    seen = set()
    deduped: List[str] = []
    for b in bullets:
        k = b.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(b)
        if len(deduped) >= cfg.max_total_bullets:
            break

    return "TEXT_FINDINGS:\n" + "\n".join([f"- {b}" for b in deduped])


# TXT facts block
def build_facts_block(
    file_path: Path,
    *,
    txt_preview_chars: int = 6000,
    deep_review_analysis: bool = False,
    deep_max_chars: int = 12000,
) -> str:
    suffix = file_path.suffix.lower()
    if suffix != ".txt":
        raise ValueError(f"build_facts_block only supports .txt, got: {suffix}")

    full = _dedupe_lines(read_txt_full(file_path))
    if not full.strip():
        return (
            "FILE_TYPE: TXT\n"
            f"FILE_NAME: {file_path.name}\n"
            "CONTENT_PREVIEW:\nINSUFFICIENT DATA\n"
        )

    if not deep_review_analysis:
        preview = full[:txt_preview_chars]
        return (
            "FILE_TYPE: TXT\n"
            f"FILE_NAME: {file_path.name}\n"
            f"CONTENT_PREVIEW:\n{preview}\n\n"
            "TEXT_FINDINGS: SKIPPED (deep_review_analysis disabled)\n"
            "REVIEW_THEME_SUMMARY: SKIPPED (deep_review_analysis disabled)\n"
        )

    src = full[:deep_max_chars]
    preview = src[:txt_preview_chars]

    findings = extract_text_findings_api(src, ChunkConfig())
    themes = summarize_reviews_themes(src)

    return (
        "FILE_TYPE: TXT\n"
        f"FILE_NAME: {file_path.name}\n"
        f"CONTENT_PREVIEW:\n{preview}\n\n"
        f"{findings}\n\n"
        "REVIEW_THEME_SUMMARY (use ONLY these themes; do not invent others):\n"
        f"{themes}\n"
    )


# CSV summarization
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

        col_asin = col_pick("asin", "ASIN")
        col_title = col_pick("title", "product_title", "name", "product_name")
        col_price = col_pick("price", "avg_price", "price_usd")
        col_rating = col_pick("rating", "stars", "avg_rating")
        col_reviews = col_pick("review_count", "reviews", "reviews_count", "rating_count")
        col_weight = col_pick("weight_lbs", "weight", "shipping_weight", "weight_lb")
        col_category = col_pick("category", "cat")

        prices, ratings, reviews, weights = [], [], [], []
        categories = set()
        asins = []
        titles = []

        row_count = 0
        for row in reader:
            row_count += 1
            if row_count > max_rows:
                break

            if col_asin:
                a = (row.get(col_asin, "") or "").strip()
                if a:
                    asins.append(a)

            if col_title:
                t = (row.get(col_title, "") or "").strip()
                if t:
                    titles.append(t)

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

    def band_label(value, low, high, low_label, mid_label, high_label) -> str:
        if value is None:
            return "INSUFFICIENT DATA"
        if value < low:
            return low_label
        if value <= high:
            return mid_label
        return high_label

    price_competition = band_label(price_spread_pct, 15.0, 30.0, "LOW (tight pricing band)", "MEDIUM", "HIGH (price war risk)")
    quality_consistency = band_label(rating_spread, 0.30, 0.60, "STABLE", "MIXED", "VOLATILE")
    review_concentration = band_label(review_dominance, 1.50, 2.50, "FRAGMENTED (no single dominant listing)", "MODERATE", "DOMINANT (top listing likely hard to beat)")
    shipping_complexity = band_label(weight_avg, 1.50, 3.00, "LIGHT", "MEDIUM", "HEAVY")

    sample_confidence = "LOW (very small sample)" if row_count < 20 else "MEDIUM/HIGH (larger sample)"

    asin_sample = ", ".join(asins[:10]) if asins else "INSUFFICIENT DATA"
    title_sample = ", ".join(titles[:5]) if titles else "INSUFFICIENT DATA"

    lines = [
        "FILE_TYPE: CSV",
        f"FILE_NAME: {path.name}",
        f"ROWS_READ: {row_count}",
        "COLUMNS:",
        ("- " + ("\n- ".join(cols) if cols else "INSUFFICIENT DATA")),
        "",
        "IDENTIFIERS:",
        f"ASIN_SAMPLE: {asin_sample}",
        f"PRODUCT_TITLE_SAMPLE: {title_sample}",
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


# Combined facts
def build_combined_facts_block(
    *,
    metrics_csv: Optional[Path] = None,
    reviews_txt: Optional[Path] = None,
    deep_review_analysis: bool = False,
    lead_time_days: int = 60,
    target_margin_pct: float = 30.0,
    ad_spend_per_unit_usd: float = 0.0,
) -> str:
    sections: list[str] = []

    def add_section(title: str, content: str):
        sections.append(f"=== {title} ===\n{content}".strip())

    # Assumptions as a citable section
    add_section(
        "ASSUMPTIONS",
        "\n".join(
            [
                "ASSUMPTIONS:",
                f"- LEAD_TIME_DAYS: {int(lead_time_days)}",
                f"- TARGET_MARGIN_PCT: {float(target_margin_pct):.2f}",
                f"- AD_SPEND_PER_UNIT_USD: {float(ad_spend_per_unit_usd):.2f}",
                "",
                "ASSUMPTIONS_RULES:",
                "- These are user-provided inputs for model-free calculations only.",
                "- Do not treat as observed market data.",
            ]
        ),
    )

    if metrics_csv is not None:
        if not metrics_csv.exists():
            raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")
        add_section("METRICS", summarize_csv(metrics_csv))

    if reviews_txt is not None:
        if not reviews_txt.exists():
            raise FileNotFoundError(f"Reviews TXT not found: {reviews_txt}")
        txt_block = build_facts_block(reviews_txt, deep_review_analysis=deep_review_analysis)
        add_section("REVIEWS", txt_block)

    if not sections:
        return "NO_INPUT_DATA: INSUFFICIENT DATA"

    return "\n\n".join(sections)