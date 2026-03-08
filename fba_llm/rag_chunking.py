from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source_id: str
    source_type: str
    location: str

def chunk_text(
    text: str,
    *,
    source_id: str,
    source_type: str,
    base_location: str,
    max_chars: int = 1100,
    overlap: int = 180,
) -> list[Chunk]:
    text = (text or "").strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 1
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        window = text[start:end].strip()

        if window:
            cid = f"{source_id}::chunk{idx}"
            loc = f"{base_location} | chunk {idx}"
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    text=window,
                    source_id=source_id,
                    source_type=source_type,
                    location=loc,
                )
            )
            idx += 1

        if end >= n:
            break
        start = max(0, end - overlap)

    return chunks