from __future__ import annotations
import re
from fba_llm.model import get_input_device
from dataclasses import dataclass
from typing import List


@dataclass
class ChunkConfig:
    chunk_size: int = 2200      # chars per chunk
    overlap: int = 200          # chars overlap
    max_chunks: int = 12
    max_bullets_per_chunk: int = 6
    max_total_bullets: int = 18


def chunk_text(text: str, chunk_size: int = 2200, overlap: int = 200) -> List[str]:
    text = text or ""
    if not text:
        return []
    overlap = max(0, min(overlap, chunk_size - 1))
    step = max(1, chunk_size - overlap)

    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        chunks.append(text[i:i + chunk_size])
        i += step
    return chunks


def select_chunks_evenly(chunks: List[str], max_chunks: int) -> List[str]:
    if len(chunks) <= max_chunks:
        return chunks
    if max_chunks <= 0:
        return []

    # Always include first and last if possible
    picks = [0]
    if max_chunks > 1:
        picks.append(len(chunks) - 1)

    # Fill remaining evenly
    remaining = max_chunks - len(picks)
    if remaining > 0:
        # indices between 1 and len-2
        span = len(chunks) - 2
        if span > 0:
            for k in range(1, remaining + 1):
                idx = 1 + round(k * (span / (remaining + 1)))
                picks.append(idx)

    picks = sorted(set(picks))
    return [chunks[i] for i in picks[:max_chunks]]


def extract_text_findings(tokenizer, model, text: str, cfg: ChunkConfig = ChunkConfig()) -> str:
    chunks = chunk_text(text, cfg.chunk_size, cfg.overlap)
    chunks = select_chunks_evenly(chunks, cfg.max_chunks)

    print(f"[chunking] total_chunks={len(chunks)}")

    if not chunks:
        return "TEXT_FINDINGS:\n- INSUFFICIENT DATA"

    all_bullets: List[str] = []

    for idx, ch in enumerate(chunks, start=1):
        print(f"[chunking] running chunk {idx}/{len(chunks)} ...")
        prompt = (
            "You are extracting evidence from customer text.\n"
            "RULES (STRICT):\n"
            "- Use ONLY the provided TEXT CHUNK.\n"
            "- Do NOT add facts not present.\n"
            "- Hyphen bullets only. No numbered lists.\n"
            "- Do NOT output any numbers unless that exact number appears in the TEXT CHUNK.\n"
            "- Focus on actionable themes: complaints, praise, defects, shipping/packaging issues, sizing/weight mentions.\n"
            "- Do NOT infer or estimate percentages/ratios/shares.\n"
            "- Use qualitative frequency words only: many/some/several/few.\n"
            f"- Output at most {cfg.max_bullets_per_chunk} bullets.\n\n"
            f"TEXT CHUNK {idx}/{len(chunks)}:\n{ch}\n\n"
            "BULLETS:\n- "
        )

        device = get_input_device(model)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        gen = outputs[0][inputs["input_ids"].shape[-1]:]
        out = tokenizer.decode(gen, skip_special_tokens=True).strip()

        # Normalize into bullets (defensive + cleanup)
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue

            # drop separator lines
            if set(line) <= {"-", "_", "="} and len(line) >= 10:
                continue

            # remove a leading "-" if present
            if line.startswith("-"):
                line = line[1:].strip()

            # remove common numbering like "1." / "2)" / "3 -"
            line = re.sub(r"^\s*\d+\s*[\.\)\-:]\s*", "", line)

            # trim weird trailing junk
            line = line.strip(" -•\t")

            # ignore very short fragments
            if len(line) < 8:
                continue

            all_bullets.append(line)

    # Dedupe while preserving order (simple)
    seen = set()
    deduped: List[str] = []
    for b in all_bullets:
        key = b.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(b)
        if len(deduped) >= cfg.max_total_bullets:
            break

    if not deduped:
        return "TEXT_FINDINGS:\n- INSUFFICIENT DATA"

    lines = ["TEXT_FINDINGS:"]
    lines += [f"- {b}" for b in deduped]
    return "\n".join(lines)
