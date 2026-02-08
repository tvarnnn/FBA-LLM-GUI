import re
from typing import Dict, List, Optional

ALLOWED_STATES = {"PROCEED", "PROCEED_WITH_CAUTION", "DO_NOT_PROCEED", "INSUFFICIENT_DATA"}

# Accept -, *, • as bullets
_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.*\S)\s*$")


def _grab_section_ci(text: str, header: str) -> str:
    """
    Case-insensitive section grabber.
    Captures everything after 'header:' until the next 'something:' header line or end.
    """
    # header label like "KEY SIGNALS" without colon
    # Stop when we hit a line that looks like a header: "Words:" (not a bullet)
    pattern = rf"(?im)^\s*{re.escape(header)}\s*:\s*(.*?)(?=^\s*[A-Za-z][A-Za-z \t]+?\s*:\s|\Z)"
    m = re.search(pattern, text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""


def _grab_any_section_ci(text: str, headers: List[str]) -> str:
    for h in headers:
        body = _grab_section_ci(text, h)
        if body:
            return body
    return ""


def _two_bullets_loose(section_body: str) -> List[str]:
    items: List[str] = []
    for line in section_body.splitlines():
        m = _BULLET_RE.match(line)
        if m:
            items.append(m.group(1).strip())

    # If the model didn't use bullets, as a fallback take non-empty lines
    if not items:
        for line in section_body.splitlines():
            s = line.strip()
            if s and not s.endswith(":"):
                items.append(s)
            if len(items) >= 2:
                break

    while len(items) < 2:
        items.append("INSUFFICIENT DATA")
    return items[:2]


def _infer_state_from_text(text: str) -> str:
    upper = text.upper()
    if "PROCEED_WITH_CAUTION" in upper or "PROCEED WITH CAUTION" in upper:
        return "PROCEED_WITH_CAUTION"
    if "DO_NOT_PROCEED" in upper or "DO NOT PROCEED" in upper:
        return "DO_NOT_PROCEED"
    # "PROCEEDED" variants
    if "PROCEEDED WITH CAUTION" in upper:
        return "PROCEED_WITH_CAUTION"
    if "PROCEEDED" in upper or "PROCEED" in upper:
        return "PROCEED"
    return "INSUFFICIENT_DATA"


def parse_advisor_text_to_json(text: str) -> Dict:
    summary_body = _grab_any_section_ci(text, ["SUMMARY"])
    key_body = _grab_any_section_ci(text, ["KEY SIGNALS", "KEY SIGNALLS", "KEY SIGNAL", "KEY SIGNALS"])
    risks_body = _grab_any_section_ci(text, ["RISKS", "RISK"])
    missing_body = _grab_any_section_ci(text, ["MISSING DATA", "MISSING", "MISSION"])

    decision_body = _grab_any_section_ci(text, ["DECISION"])

    summary = _two_bullets_loose(summary_body)
    key_signals = _two_bullets_loose(key_body)
    risks = _two_bullets_loose(risks_body)
    missing_data = _two_bullets_loose(missing_body)

    # State: prefer explicit STATE: line if present; otherwise infer
    state = "INSUFFICIENT_DATA"
    m = re.search(r"(?im)^\s*STATE\s*:\s*([A-Z_]+)\s*$", decision_body)
    if m and m.group(1).strip() in ALLOWED_STATES:
        state = m.group(1).strip()
    else:
        state = _infer_state_from_text(text if not decision_body else decision_body)

    # RATIONALE / NEXT ACTIONS if present
    rationale_body = ""
    m = re.search(r"(?is)RATIONALE\s*:\s*(.*?)(?=\n\s*NEXT ACTIONS\s*:|\Z)", decision_body)
    if m:
        rationale_body = m.group(1).strip()

    next_actions_body = ""
    m = re.search(r"(?is)NEXT ACTIONS\s*:\s*(.*)$", decision_body)
    if m:
        next_actions_body = m.group(1).strip()

    rationale = _two_bullets_loose(rationale_body)
    next_actions = _two_bullets_loose(next_actions_body)

    return {
        "summary": summary,
        "key_signals": key_signals,
        "risks": risks,
        "missing_data": missing_data,
        "decision": {
            "state": state,
            "rationale": rationale,
            "next_actions": next_actions,
        },
    }
