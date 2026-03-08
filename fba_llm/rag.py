from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fba_llm.vectorstore import query_chunks
from fba_llm.rag_chunking import Chunk

@dataclass(frozen=True)
class Evidence:
    key: str        # A, B, C...
    source_id: str
    source_type: str
    location: str
    text: str

def _letters():
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        yield c

def _dedupe_by_id(chunks: list[Chunk]) -> list[Chunk]:
    seen = set()
    out = []
    for ch in chunks:
        if ch.chunk_id in seen:
            continue
        seen.add(ch.chunk_id)
        out.append(ch)
    return out

def retrieve_evidence(chroma_dir: Path, question: str, *, k_total: int = 12) -> list[Evidence]:
    # Multi-query “operator probes” so we don't miss failure signals.
    probes = [
        question,
        "durability break broken defect material quality control",
        "packaging shipping damage missing parts",
        "value for money cheap flimsy",
        "instructions confusing assembly difficult",
    ]

    gathered: list[Chunk] = []
    for q in probes:
        gathered.extend(query_chunks(chroma_dir, q, k=6))

    gathered = _dedupe_by_id(gathered)
    gathered = gathered[:k_total]

    out: list[Evidence] = []
    for letter, ch in zip(_letters(), gathered):
        txt = (ch.text or "").strip()
        if len(txt) > 1100:
            txt = txt[:1100] + "..."
        out.append(
            Evidence(
                key=letter,
                source_id=ch.source_id or "unknown",
                source_type=ch.source_type or "unknown",
                location=ch.location or "unknown",
                text=txt,
            )
        )
    return out

def format_context_snippets(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(no retrieved context)"
    blocks = []
    for e in evidence:
        blocks.append(f"[{e.key}] TYPE={e.source_type} SOURCE={e.source_id} | LOC={e.location}\n{e.text}")
    return "\n\n".join(blocks)