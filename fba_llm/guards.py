from __future__ import annotations

import re
from typing import Iterable


# Numbers that are usually harmless formatting/list markers.
# Tweak this set if you want stricter behavior.
ALLOWED_FORMAT_NUMBERS = {"1", "2", "3", "4", "5"}


def extract_numbers(text: str, *, allow_format_numbers: Iterable[str] = ALLOWED_FORMAT_NUMBERS) -> set[str]:
    """
    Extract numeric-like tokens from text.
    Normalizes by removing '$' and ',' and trimming trailing punctuation.

    allow_format_numbers:
      numbers that should be ignored as likely list markers (e.g., 1,2,3).
    """
    # Matches: 1, 12, 1.23, $12.34, 1,234.56, etc.
    pattern = r"\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\$?\d+(?:\.\d+)?"
    found = re.findall(pattern, text or "")

    allow = set(str(x) for x in allow_format_numbers)

    norm: set[str] = set()
    for s in found:
        s2 = s.strip().rstrip(".,);:%]}")

        # normalize currency and commas
        s2 = s2.replace("$", "").replace(",", "").strip()

        if not s2:
            continue

        # Ignore simple list-marker numbers like "1", "2", "3"
        if s2 in allow:
            continue

        norm.add(s2)

    return norm


def check_no_new_numbers(output_text: str, input_text: str) -> tuple[bool, set[str]]:
    """
    Returns (ok, extras).
    ok=True when every number in output also appears in input (after normalization),
    except for allowed formatting numbers (1..5 by default).
    """
    out_nums = extract_numbers(output_text)
    in_nums = extract_numbers(input_text)

    extras = {n for n in out_nums if n not in in_nums}
    return (len(extras) == 0), extras


def check_no_banned_claims(output_text: str, input_text: str) -> tuple[bool, list[str]]:
    out = (output_text or "").lower()
    src = (input_text or "").lower()

    banned_keywords = [
        "bsr",
        "best seller rank",
        "sales volume",
        "sales velocity",
        "units sold",
        "selling well",
        "time on market",
        "listing age",
        "ppc",
        "ad spend",
        "profit margin",
        "cogs",
        "landed cost",
        "return rate",
    ]

    hits = []
    for kw in banned_keywords:
        if kw in out and kw not in src:
            hits.append(kw)

    return (len(hits) == 0), hits