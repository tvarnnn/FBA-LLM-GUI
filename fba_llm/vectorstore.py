from __future__ import annotations

import warnings
from dataclasses import asdict
from pathlib import Path
from typing import List

from fba_llm.rag_chunking import Chunk

COLLECTION_NAME = "fba_chunks"

def _get_embed_fn():
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction  # type: ignore
    return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def _ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _fallback_jsonl_path(persist_dir: Path) -> Path:
    persist_dir = _ensure_dir(persist_dir)
    return persist_dir / "chunks.jsonl"

def _fallback_upsert(persist_dir: Path, chunks: List[Chunk]) -> int:
    out = _fallback_jsonl_path(persist_dir)
    import json
    n_written = 0
    with out.open("a", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            n_written += 1
    return n_written

def upsert_chunks(persist_dir: Path, chunks: List[Chunk]) -> int:
    persist_dir = Path(persist_dir)
    if not chunks:
        _ensure_dir(persist_dir)
        return 0

    try:
        import chromadb
        from chromadb.config import Settings

        embed_fn = _get_embed_fn()
        _ensure_dir(persist_dir)

        client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
        )

        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [
            {
                "source_id": c.source_id,
                "source_type": c.source_type,
                "location": c.location,
            }
            for c in chunks
        ]

        # upsert if available
        if hasattr(collection, "upsert"):
            collection.upsert(ids=ids, documents=docs, metadatas=metas)
        else:
            try:
                collection.delete(ids=ids)
            except Exception:
                pass
            collection.add(ids=ids, documents=docs, metadatas=metas)

        return len(chunks)

    except Exception as e:
        warnings.warn(f"chromadb upsert failed, falling back to JSONL: {e}")
        return _fallback_upsert(persist_dir, chunks)

def query_chunks(persist_dir: Path, query: str, k: int = 8) -> List[Chunk]:
    persist_dir = Path(persist_dir)

    try:
        import chromadb
        from chromadb.config import Settings

        embed_fn = _get_embed_fn()
        client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
        )

        res = collection.query(query_texts=[query], n_results=k, include=["documents", "metadatas", "ids"])
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]

        out: List[Chunk] = []
        for cid, doc, meta in zip(ids, docs, metas):
            meta = meta or {}
            out.append(
                Chunk(
                    chunk_id=str(cid),
                    text=str(doc or ""),
                    source_id=str(meta.get("source_id", "")),
                    source_type=str(meta.get("source_type", "")),
                    location=str(meta.get("location", "")),
                )
            )
        return out

    except Exception as e:
        warnings.warn(f"chromadb query failed, falling back to JSONL: {e}")
        # fallback: return last k lines
        p = _fallback_jsonl_path(persist_dir)
        if not p.exists():
            return []
        import json
        lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        tail = lines[-k:]
        out: List[Chunk] = []
        for ln in tail:
            try:
                obj = json.loads(ln)
                out.append(Chunk(**obj))
            except Exception:
                continue
        return out