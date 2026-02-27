from __future__ import annotations

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
        span = len(chunks) - 2
        if span > 0:
            for k in range(1, remaining + 1):
                idx = 1 + round(k * (span / (remaining + 1)))
                picks.append(idx)

    picks = sorted(set(picks))
    return [chunks[i] for i in picks[:max_chunks]]