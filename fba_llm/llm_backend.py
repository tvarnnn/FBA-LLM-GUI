from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Dict, Tuple, List

from dotenv import load_dotenv
from langchain_groq import ChatGroq


# .env loading (reliable)
def _load_env_once() -> None:
    here = Path(__file__).resolve()
    project_root = here.parents[1]  # fba_llm/.. (project root)
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


_load_env_once()


class LlmError(RuntimeError):
    pass


def _require_env(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise LlmError(f"Missing required env var: {name}")
    return v


def _is_debug() -> bool:
    return (os.getenv("LLM_DEBUG") or "").strip().lower() in {"1", "true", "yes", "on"}


# Cache clients by (model, temperature, max_tokens, timeout_s)
_CLIENT_CACHE: Dict[Tuple[str, float, int, int], ChatGroq] = {}


def _get_groq_client(model: str, temperature: float, max_tokens: int, timeout_s: int) -> ChatGroq:
    key = (model, float(temperature), int(max_tokens), int(timeout_s))
    if key in _CLIENT_CACHE:
        return _CLIENT_CACHE[key]

    # Force-check so failures are obvious
    _require_env("GROQ_API_KEY")

    # langchain_groq supports request_timeout in newer versions; harmless if ignored
    client = ChatGroq(
        model=model,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        request_timeout=int(timeout_s),
    )

    _CLIENT_CACHE[key] = client
    return client


# Public API
def generate_text(
    prompt: str,
    *,
    provider: Optional[str] = None,
    model_override: Optional[str] = None,
    max_tokens: int = 500,
    temperature: float = 0.0,
    timeout_s: int = 60,
) -> str:
    p = (provider or os.getenv("LLM_PROVIDER") or "groq").strip().lower()
    if p != "groq":
        raise LlmError(f"Provider {p!r} not supported in this backend. Use groq for now.")

    # Default to the strong reasoning model
    default_model = (os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip()
    model = (model_override or default_model).strip()

    llm = _get_groq_client(
        model=model,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        timeout_s=int(timeout_s),
    )

    if _is_debug():
        print(f"[LLM] provider=groq model={model} max_tokens={max_tokens} temperature={temperature} timeout_s={timeout_s}")

    t0 = time.perf_counter()
    try:
        resp = llm.invoke(prompt)
        text = (getattr(resp, "content", "") or "").strip()
    except Exception as e:
        raise LlmError(f"Groq call failed (model={model}): {e}") from None
    finally:
        if _is_debug():
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"[LLM] done model={model} latency_ms={dt:.0f}")

    return text


def list_models(provider: str = "groq") -> List[str]:
    p = (provider or "").strip().lower()
    if p != "groq":
        raise LlmError(f"Unknown provider: {provider!r}")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # no key loaded yet; return common ids
        return [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "groq/compound-mini",
        ]

    url = "https://api.groq.com/openai/v1/models"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw) if raw else {}
        ids = []
        for item in (data.get("data") or []):
            mid = item.get("id")
            if mid:
                ids.append(mid)
        ids = sorted(set(ids))
        return ids or [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "groq/compound-mini",
        ]
    except Exception:
        # fallback list (don’t crash your app because list models failed)
        return [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "groq/compound-mini",
        ]