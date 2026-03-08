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


# .env loading
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


# Groq client cache
_CLIENT_CACHE: Dict[Tuple[str, float, int, int], ChatGroq] = {}


def _get_groq_client(model: str, temperature: float, max_tokens: int, timeout_s: int) -> ChatGroq:
    key = (model, float(temperature), int(max_tokens), int(timeout_s))
    if key in _CLIENT_CACHE:
        return _CLIENT_CACHE[key]

    _require_env("GROQ_API_KEY")

    client = ChatGroq(
        model=model,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        request_timeout=int(timeout_s),
    )
    _CLIENT_CACHE[key] = client
    return client


# Provider helpers
def _resolve_provider(provider: Optional[str] = None) -> str:
    p = (provider or os.getenv("LLM_PROVIDER") or "groq").strip().lower()
    aliases = {
        "claude": "anthropic",
    }
    return aliases.get(p, p)


def _default_model_for_provider(provider: str) -> str:
    if provider == "groq":
        return (os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip()
    if provider == "anthropic":
        return (os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-4-0").strip()
    raise LlmError(f"Unsupported provider: {provider!r}")


# Groq text generation
def _generate_text_groq(
    prompt: str,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
) -> str:
    llm = _get_groq_client(
        model=model,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        timeout_s=int(timeout_s),
    )

    if _is_debug():
        print(
            f"[LLM] provider=groq model={model} "
            f"max_tokens={max_tokens} temperature={temperature} timeout_s={timeout_s}"
        )

    t0 = time.perf_counter()
    try:
        resp = llm.invoke(prompt)
        text = (getattr(resp, "content", "") or "").strip()
    except Exception as e:
        raise LlmError(f"Groq call failed (model={model}): {e}") from None
    finally:
        if _is_debug():
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"[LLM] done provider=groq model={model} latency_ms={dt:.0f}")

    return text


# Anthropic text generation
def _generate_text_anthropic(
    prompt: str,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
) -> str:
    api_key = _require_env("ANTHROPIC_API_KEY")

    url = "https://api.anthropic.com/v1/messages"
    body = {
        "model": model,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )

    if _is_debug():
        print(
            f"[LLM] provider=anthropic model={model} "
            f"max_tokens={max_tokens} temperature={temperature} timeout_s={timeout_s}"
        )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=int(timeout_s)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(e)
        raise LlmError(f"Anthropic call failed (model={model}): {detail}") from None
    except Exception as e:
        raise LlmError(f"Anthropic call failed (model={model}): {e}") from None
    finally:
        if _is_debug():
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"[LLM] done provider=anthropic model={model} latency_ms={dt:.0f}")

    parts = data.get("content") or []
    text_chunks: List[str] = []
    for item in parts:
        if isinstance(item, dict) and item.get("type") == "text":
            text_chunks.append((item.get("text") or "").strip())

    text = "\n".join([x for x in text_chunks if x]).strip()
    return text


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
    p = _resolve_provider(provider)
    model = (model_override or _default_model_for_provider(p)).strip()

    if p == "groq":
        return _generate_text_groq(
            prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )

    if p == "anthropic":
        return _generate_text_anthropic(
            prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )

    raise LlmError(f"Provider {p!r} not supported. Supported: groq, anthropic")


def list_models(provider: str = "groq") -> List[str]:
    p = _resolve_provider(provider)

    if p == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
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
            return [
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
                "groq/compound-mini",
            ]

    if p == "anthropic":
        # The actual model used should come from ANTHROPIC_MODEL or model_override.
        env_model = (os.getenv("ANTHROPIC_MODEL") or "").strip()
        out = []
        if env_model:
            out.append(env_model)

        # Sensible fallback suggestions for UI convenience.
        out.extend([
            "claude-sonnet-4-0",
            "claude-3-7-sonnet-latest",
            "claude-3-5-sonnet-latest",
        ])

        # de-dupe while preserving order
        seen = set()
        deduped = []
        for m in out:
            if m and m not in seen:
                seen.add(m)
                deduped.append(m)
        return deduped

    raise LlmError(f"Unknown provider: {provider!r}")