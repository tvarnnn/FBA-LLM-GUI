from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, List


# -----------------------------
# .env loading (reliable)
# -----------------------------
def _load_env_once() -> None:
    """
    Load .env from project root reliably.
    Project root assumed to be parent of this package folder:
      project_root/
        gui_app.py
        .env
        fba_llm/
          llm_backend.py  <-- this file
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    here = Path(__file__).resolve()
    project_root = here.parents[1]  # fba_llm/.. (project root)
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # fallback: try current working dir
        load_dotenv()


_load_env_once()


class LlmError(RuntimeError):
    pass


def _require_env(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise LlmError(f"Missing required env var: {name}")
    return v


def _http_json(
    req: urllib.request.Request,
    *,
    timeout_s: int = 60,
) -> dict:
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        # IMPORTANT: read body so you can see the actual error
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise LlmError(f"HTTPError {e.code}: {body or '(no body)'}") from None
    except Exception as e:
        raise LlmError(f"Request failed: {e}") from None


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------
def generate_text(
    prompt: str,
    *,
    provider: Optional[str] = None,
    model_override: Optional[str] = None,
    max_tokens: int = 400,
    temperature: float = 0.2,
    timeout_s: int = 60,
) -> str:
    """
    provider: "groq" or "anthropic" (aka "claude")
    model_override: optional model string set from GUI
    """
    p = (provider or os.getenv("LLM_PROVIDER") or "groq").strip().lower()

    if p == "groq":
        return _groq_generate(
            prompt,
            model_override=model_override,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )

    if p in ("anthropic", "claude"):
        return _anthropic_generate(
            prompt,
            model_override=model_override,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )

    raise LlmError(f"Unknown provider: {p!r}")


def list_models(provider: str) -> List[str]:
    """
    Returns list of model ids visible to this API key.
    Use this to stop guessing and verify access.
    """
    p = (provider or "").strip().lower()
    if p == "groq":
        return _groq_list_models()
    if p in ("anthropic", "claude"):
        # Anthropic doesn’t provide a simple public "list models" endpoint like OpenAI-compatible.
        # Return the env default / common names.
        return [
            os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
        ]
    raise LlmError(f"Unknown provider: {provider!r}")


# ---------------------------------------------------------
# Groq (OpenAI-compatible)
# ---------------------------------------------------------
def _groq_generate(
    prompt: str,
    *,
    model_override: Optional[str],
    max_tokens: int,
    temperature: float,
    timeout_s: int,
) -> str:
    api_key = _require_env("GROQ_API_KEY")

    # Use a SAFE default that usually exists on free keys.
    default_model = os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant"
    model = (model_override or default_model).strip()

    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    data = _http_json(req, timeout_s=timeout_s)

    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        raise LlmError(f"Groq unexpected response: {data}") from None


def _groq_list_models() -> List[str]:
    api_key = _require_env("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/models"

    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )

    data = _http_json(req, timeout_s=30)
    models = []
    for item in data.get("data", []) or []:
        mid = item.get("id")
        if mid:
            models.append(mid)
    return sorted(set(models))


# ---------------------------------------------------------
# Anthropic
# ---------------------------------------------------------
def _anthropic_generate(
    prompt: str,
    *,
    model_override: Optional[str],
    max_tokens: int,
    temperature: float,
    timeout_s: int,
) -> str:
    api_key = _require_env("ANTHROPIC_API_KEY")
    default_model = os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest"
    model = (model_override or default_model).strip()

    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": model,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "messages": [{"role": "user", "content": prompt}],
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    data = _http_json(req, timeout_s=timeout_s)

    try:
        blocks = data.get("content", []) or []
        texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
        return ("\n".join(texts)).strip()
    except Exception:
        raise LlmError(f"Anthropic unexpected response: {data}") from None