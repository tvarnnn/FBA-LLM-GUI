import re


def extract_numbers(text: str) -> set[str]:
    pattern = r"\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\$?\d+(?:\.\d+)?"
    found = re.findall(pattern, text)
    norm: set[str] = set()
    for s in found:
        s2 = s.strip().rstrip(".,);:%]}")
        s2 = s2.replace("$", "").replace(",", "")
        if s2:
            norm.add(s2)
    return norm


def check_no_new_numbers(output_text: str, input_text: str) -> tuple[bool, set[str]]:
    out_nums = extract_numbers(output_text)
    in_nums = extract_numbers(input_text)
    extras = {n for n in out_nums if n not in in_nums}
    return (len(extras) == 0), extras


def check_no_banned_claims(output_text: str, input_text: str) -> tuple[bool, list[str]]:
    """
    Conservative guard: block mentions of metrics you *didn't* provide.
    We allow those terms only if they appear in INPUT DATA (columns/content).
    """
    out = output_text.lower()
    src = input_text.lower()

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
