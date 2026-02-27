from __future__ import annotations

import re
from typing import Iterable


# Formatting numbers allowed
# These are usually list markers or harmless UI artifacts.
ALLOWED_FORMAT_NUMBERS = {"1", "2", "3", "4", "5"}


# Number word handling
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand",
    "half", "double", "twice", "dozen", "couple",
    "majority", "minority"
}


# Numeric extraction (digits)
def extract_numbers(
    text: str,
    *,
    allow_format_numbers: Iterable[str] = ALLOWED_FORMAT_NUMBERS
) -> set[str]:
    if not text:
        return set()

    # Matches:
    #  1
    #  12
    #  1.23
    #  $12.34
    #  1,234.56
    pattern = r"\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\$?\d+(?:\.\d+)?"
    found = re.findall(pattern, text)

    allow = {str(x) for x in allow_format_numbers}
    norm: set[str] = set()

    for raw in found:
        s = raw.strip().rstrip(".,);:%]}")

        # Normalize currency + commas
        s = s.replace("$", "").replace(",", "").strip()

        if not s:
            continue

        # Ignore simple list-marker numbers
        if s in allow:
            continue

        norm.add(s)

    return norm


# Number word extraction
def extract_number_words(text: str) -> set[str]:
    if not text:
        return set()

    words = re.findall(r"[A-Za-z]+", text)
    return {w.lower() for w in words if w.lower() in NUMBER_WORDS}


# Guard: no new numbers (digits OR words)
def check_no_new_numbers(output_text: str, input_text: str) -> tuple[bool, set[str]]:
    out_nums = extract_numbers(output_text)
    in_nums = extract_numbers(input_text)

    out_words = extract_number_words(output_text)
    in_words = extract_number_words(input_text)

    extras = set()

    extras |= {n for n in out_nums if n not in in_nums}
    extras |= {w for w in out_words if w not in in_words}

    return (len(extras) == 0), extras


# Guard: banned business claims
def check_no_banned_claims(output_text: str, input_text: str) -> tuple[bool, list[str]]:
    out = (output_text or "").lower()
    src = (input_text or "").lower()

    banned_keywords = [
        # Amazon-specific metrics
        "bsr",
        "best seller rank",

        # Sales / demand claims
        "sales volume",
        "sales velocity",
        "units sold",
        "selling well",

        # Time / listing claims
        "time on market",
        "listing age",

        # Advertising
        "ppc",
        "ad spend",

        # Economics
        "profit margin",
        "margin",
        "cogs",
        "landed cost",

        # Returns (plural + variants)
        "return rate",
        "return rates"
    ]

    hits: list[str] = []
    for kw in banned_keywords:
        if kw in out and kw not in src:
            hits.append(kw)

    return (len(hits) == 0), hits