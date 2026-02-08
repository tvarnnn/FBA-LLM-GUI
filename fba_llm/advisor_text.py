from typing import Set

ALLOWED_STATES: Set[str] = {"PROCEED", "PROCEED_WITH_CAUTION", "DO_NOT_PROCEED", "INSUFFICIENT_DATA"}


def build_prompt(question: str, facts_block: str) -> str:
    return (
        "You are an Amazon FBA decision-support assistant.\n\n"

        "HARD RULES:\n"
        "- Use ONLY information from INPUT DATA.\n"
        "- Do NOT invent facts.\n"
        "- Do NOT output ANY numbers unless that exact number appears in INPUT DATA.\n"
        "- Do NOT invent percentages, ratios, or shares.\n"
        "- If describing frequency, use ONLY: many / some / several / few.\n"
        "- Do NOT use numeric words like half, double, twice, majority, minority.\n"
        "- No code blocks, no markdown fences, no examples.\n\n"

        "OUTPUT FORMAT (exact headers, hyphen bullets only):\n\n"

        "CURRENT SNAPSHOT:\n"
        "- <bullet>\n"
        "- <bullet>\n\n"

        "DIRECTIONAL FORECAST (QUALITATIVE, CONDITIONAL):\n"
        "- <if/then bullet>\n"
        "- <if/then bullet>\n\n"

        "RISKS / UNKNOWN:\n"
        "- <bullet>\n"
        "- <bullet>\n\n"

        "NEXT ACTIONS:\n"
        "- <bullet>\n"
        "- <bullet>\n\n"

        f"INPUT DATA:\n{facts_block}\n\n"
        f"QUESTION:\n{question}\n\n"

        "ANSWER:\n"
    )


def run_advisor_text(tokenizer, model, question: str, facts_block: str) -> str:
    prompt = build_prompt(question, facts_block)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=360,
        min_new_tokens=40,
        do_sample=False,
        repetition_penalty=1.08,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    gen = outputs[0][inputs["input_ids"].shape[-1]:]
    tail = tokenizer.decode(gen, skip_special_tokens=True).strip()

    answer_template = prompt.split("ANSWER:\n", 1)[1]
    return answer_template + tail
