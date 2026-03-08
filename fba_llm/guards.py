from __future__ import annotations

import re
from typing import Iterable, Set, Tuple


def check_no_new_numbers(text: str, facts_block: str) -> Tuple[bool, Set[str]]:
    """
    Your existing implementation likely lives here already.
    Keep yours if you want. If you removed the strict number guard,
    you can delete this, but advisor_text.py below no longer requires it.
    """
    # --- KEEP YOUR CURRENT IMPLEMENTATION ---
    # Placeholder safe pass-through:
    return True, set()


def check_no_banned_claims(text: str, facts_block: str) -> Tuple[bool, list[str]]:
    """
    Your existing implementation likely lives here already.
    Keep yours.
    """
    # --- KEEP YOUR CURRENT IMPLEMENTATION ---
    # Placeholder safe pass-through:
    return True, []


# ----------------------------
# NEW: Citation label validator
# ----------------------------

_CIT_PAREN_RE = re.compile(r"\(([A-Z0-9_]+)\)")
_CIT_EVIDENCE_RE = re.compile(r"(?im)\bEVIDENCE\s*:\s*(.+?)\s*$")
_CIT_TRAILING_RE = re.compile(r"(?im)^-\s.*?\s+\[([A-Z0-9_]+)\]\s*$")


def _split_labels(blob: str) -> list[str]:
    out: list[str] = []
    if not blob:
        return out
    # split on commas primarily
    for part in blob.split(","):
        s = part.strip()
        if not s:
            continue
        # allow accidental extra words; keep only label-like tokens
        # e.g. "PRICE_COMPETITION" ok, "PRICE competition" ignored
        m = re.match(r"^[A-Z0-9_]+$", s)
        if m:
            out.append(s)
    return out


def extract_citation_labels(text: str) -> Set[str]:
    """
    Supports these citation styles in output:
      - "... (LABEL)"                          -> (LABEL)
      - "... EVIDENCE: LABEL1, LABEL2"         -> EVIDENCE: ...
      - "- bullet text ... [LABEL]"            -> [LABEL] (recommended)
    """
    t = text or ""
    found: Set[str] = set()

    # (LABEL)
    for m in _CIT_PAREN_RE.finditer(t):
        found.add(m.group(1).strip())

    # EVIDENCE: A, B
    for m in _CIT_EVIDENCE_RE.finditer(t):
        labels_blob = (m.group(1) or "").strip()
        for lab in _split_labels(labels_blob):
            found.add(lab)

    # trailing [LABEL]
    for m in _CIT_TRAILING_RE.finditer(t):
        found.add(m.group(1).strip())

    return found


def check_citation_labels(text: str, allowed: Iterable[str]) -> Tuple[bool, Set[str]]:
    allowed_set = set(allowed)
    used = extract_citation_labels(text)
    invalid = {lab for lab in used if lab not in allowed_set}
    return (len(invalid) == 0), invalid