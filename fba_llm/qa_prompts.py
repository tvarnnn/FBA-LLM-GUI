from __future__ import annotations


def build_screening_summary_prompt(question: str, facts_block: str) -> str:
    q = (question or "").strip() or "Analyze this for FBA viability."

    return (
        "You are an Amazon FBA screening assistant.\n"
        "Your job is to give a cautious early-stage screening summary based ONLY on the provided evidence.\n\n"
        "RULES:\n"
        "- Use ONLY the INPUT DATA.\n"
        "- Do NOT invent sales, BSR, growth, PPC, profit, margin, COGS, fees, or demand metrics unless explicitly present.\n"
        "- If evidence is missing, say so clearly.\n"
        "- Be concise, practical, and evidence-bounded.\n"
        "- Do NOT produce a long memo.\n"
        "- Do NOT use markdown fences.\n"
        "- Hyphen bullets only.\n\n"
        "OUTPUT FORMAT:\n"
        "SCREENING SUMMARY:\n"
        "- Overall screen: <one sentence>\n"
        "- Strongest positive: <one sentence>\n"
        "- Strongest negative: <one sentence>\n"
        "- Biggest unknown: <one sentence>\n"
        "- Best next step: <one sentence>\n\n"
        "INPUT DATA:\n"
        f"{facts_block}\n\n"
        "USER GOAL:\n"
        f"{q}\n\n"
        "ANSWER:\n"
    )


def build_followup_question_prompt(user_question: str, facts_block: str) -> str:
    q = (user_question or "").strip() or "What are the biggest risks?"

    return (
        "You are an Amazon FBA research copilot.\n"
        "Answer the user's question using ONLY the provided evidence.\n\n"
        "RULES:\n"
        "- Use ONLY INPUT DATA.\n"
        "- If the question cannot be answered from the data, say exactly what is missing.\n"
        "- Do NOT invent sales, growth, BSR, margin, PPC, profit, COGS, fees, or demand metrics unless explicitly present.\n"
        "- Keep the answer practical, direct, and grounded.\n"
        "- Hyphen bullets only.\n"
        "- End with a short confidence line.\n"
        "- Do NOT use markdown fences.\n\n"
        "OUTPUT FORMAT:\n"
        "ANSWER:\n"
        "- Direct answer: <answer>\n"
        "- Evidence used: <what in the data supports this>\n"
        "- Limitation: <missing data or uncertainty, if relevant>\n"
        "Confidence: <low|medium|high>\n\n"
        "INPUT DATA:\n"
        f"{facts_block}\n\n"
        "QUESTION:\n"
        f"{q}\n\n"
        "ANSWER:\n"
    )