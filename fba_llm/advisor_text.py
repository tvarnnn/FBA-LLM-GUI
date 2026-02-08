from typing import Set

ALLOWED_STATES: Set[str] = {"PROCEED", "PROCEED_WITH_CAUTION", "DO_NOT_PROCEED", "INSUFFICIENT_DATA"}


def build_prompt(question: str, facts_block: str) -> str:
    return (
        "You are an Amazon FBA decision-support assistant.\n"
        "You must fill in the template below EXACTLY.\n"
        "Replace <line> with your content.\n"
        "Do not remove headers.\n"
        "Use hyphen '-' bullets only.\n"
        "Do NOT output code blocks, markdown, examples, or ASCII art.\n\n"

        "GROUNDING RULES:\n"
        "- Use ONLY information that appears in INPUT DATA.\n"
        "- If a fact is missing, list it under MISSING DATA.\n"
        "- Do not estimate, infer, or guess.\n\n"

        "NUMBERS RULES:\n"
        "- You may ONLY use numbers that appear verbatim in INPUT DATA.\n"
        "- Do NOT invent percentages, ratios, or shares.\n"
        "- NEVER summarize customer feedback using numbers.\n"
        "- Use ONLY qualitative terms: many / some / several / few.\n\n"

        f"INPUT DATA:\n{facts_block}\n\n"
        f"QUESTION:\n{question}\n\n"

        "ANSWER:\n"
        "SUMMARY:\n"
        "- <line>\n"
        "- <line>\n\n"
        "KEY SIGNALS:\n"
        "- <line>\n"
        "- <line>\n\n"
        "RISKS:\n"
        "- <line>\n"
        "- <line>\n\n"
        "MISSING DATA:\n"
        "- <line>\n"
        "- <line>\n\n"
        "DECISION:\n"
        "STATE: <ONE OF: PROCEED, PROCEED_WITH_CAUTION, DO_NOT_PROCEED, INSUFFICIENT_DATA>\n"
        "RATIONALE:\n"
        "- <line>\n"
        "- <line>\n"
        "NEXT ACTIONS:\n"
        "- <line>\n"
        "- <line>\n\n"
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
