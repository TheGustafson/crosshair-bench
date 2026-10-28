"""
Zero-dependency OpenRouter client for Crosshair-Bench.

Supports:
- Model specs with optional reasoning effort (`provider/model@effort`)
- `reasoning: {"effort": ...}` with automatic parameter fallback on HTTP 400
- Extraction of native `reasoning` / `reasoning_content` / `reasoning_details`
- Full per-call telemetry (`latency_ms`, `prompt_tokens`, `completion_tokens`,
  `reasoning_tokens`, `cost_usd`, `provider_model`, `finish_reason`, `raw_response_excerpt`)
"""

from __future__ import annotations

import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".config" / "crosshair-arena"
DEFAULT_KEY_FILE = CONFIG_DIR / "openrouter.env"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"

VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "none")


def ensure_secure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(CONFIG_DIR, 0o700)
    except OSError:
        pass


def load_openrouter_api_key(key_file: Path = DEFAULT_KEY_FILE) -> str:
    """
    Load OPENROUTER_API_KEY from environment or ~/.config/crosshair-arena/openrouter.env.
    Never stores or reads keys from inside the repository.
    """
    ensure_secure_config_dir()
    env_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_key:
        return env_key

    if not key_file.exists():
        raise RuntimeError(
            f"OpenRouter API key not found.\n"
            f"Store your key outside all Git repos in:\n"
            f"  {key_file}\n"
            f"or export OPENROUTER_API_KEY in your shell."
        )

    try:
        mode = key_file.stat().st_mode & 0o777
        if mode & 0o077:
            os.chmod(key_file, 0o600)
    except OSError:
        pass

    for raw_line in key_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            if k.strip() == "OPENROUTER_API_KEY":
                val = v.strip().strip("'").strip('"')
                if val:
                    return val
        elif line.startswith("sk-or-"):
            return line.strip("'").strip('"')

    raise RuntimeError(f"Could not parse OPENROUTER_API_KEY from {key_file}.")


def parse_model_spec(spec: str) -> tuple[str, str | None]:
    """
    Parse a model spec string with optional `@effort` suffix.
    Examples:
      'google/gemini-3.8-flash' -> ('google/gemini-3.8-flash', None)
      'google/gemini-4-argon@medium' -> ('google/gemini-4-argon', 'medium')
      'openai/gpt-6.1-terra-pro@high' -> ('openai/gpt-6.1-terra-pro', 'high')
    """
    raw = spec.strip()
    if "@" in raw:
        model_id, effort = raw.rsplit("@", 1)
        effort = effort.strip().lower()
        if effort not in VALID_REASONING_EFFORTS:
            raise ValueError(
                f"Invalid reasoning effort '{effort}' in '{spec}'. "
                f"Valid: {VALID_REASONING_EFFORTS}"
            )
        return model_id.strip(), effort
    return raw, None


def model_short_label(model_id: str, reasoning_effort: str | None = None) -> str:
    """Create a concise human-readable label for a model and optional reasoning effort."""
    base = model_id.split("/")[-1].lower()
    base = re.sub(r"[^a-z0-9._-]+", "-", base).strip("-")
    if reasoning_effort and reasoning_effort != "none":
        return f"{base}@{reasoning_effort}"
    return base


def _extract_native_reasoning(msg: dict[str, Any]) -> str | None:
    """Extract native reasoning text or summary from an OpenRouter message object."""
    for key in ("reasoning", "reasoning_content"):
        val = msg.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    details = msg.get("reasoning_details")
    if isinstance(details, list):
        parts: list[str] = []
        for item in details:
            if isinstance(item, dict):
                txt = item.get("text") or item.get("summary")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt.strip())
        if parts:
            return "\n".join(parts)
    return None


def call_openrouter_chat(
    *,
    api_key: str,
    model_id: str,
    messages: list[dict[str, str]],
    max_tokens: int = 4096,
    temperature: float | None = 0.0,
    reasoning_effort: str | None = None,
    timeout_s: float = 150.0,
) -> dict[str, Any]:
    """
    Execute a synchronous OpenRouter chat completion call with automatic fallback
    if a model/provider rejects `reasoning` or `temperature`.
    """
    base_payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "usage": {"include": True},
    }

    # OpenAI GPT-5/6/o-series and Anthropic Claude 5.x models often reject temperature
    # or do not list temperature in supported_parameters when reasoning is active.
    supports_temp = not model_id.startswith(
        ("openai/gpt-5", "openai/gpt-6", "openai/o", "anthropic/")
    )
    if temperature is not None and supports_temp and not reasoning_effort:
        base_payload["temperature"] = temperature

    if reasoning_effort:
        if reasoning_effort == "none":
            base_payload["reasoning"] = {"effort": "none", "exclude": True}
        else:
            base_payload["reasoning"] = {"effort": reasoning_effort}
    else:
        base_payload["include_reasoning"] = True

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "HTTP-Referer": "https://thegustafson.com",
        "X-Title": "Crosshair-Bench",
    }
    ctx = ssl.create_default_context()

    payload = dict(base_payload)
    last_err: Exception | None = None

    for attempt in range(4):
        body = json.dumps(payload).encode("utf-8")
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                OPENROUTER_CHAT_URL,
                data=body,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:
                raw = resp.read().decode("utf-8")
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            data = json.loads(raw)

            if "error" in data and data["error"]:
                err_obj = data["error"]
                err_msg = (
                    err_obj.get("message", str(err_obj))
                    if isinstance(err_obj, dict)
                    else str(err_obj)
                )
                # If provider rejected reasoning/temperature inside a 200 error envelope, strip & retry
                if attempt < 2 and (
                    "reasoning" in err_msg.lower() or "temperature" in err_msg.lower()
                ):
                    payload.pop("reasoning", None)
                    payload.pop("include_reasoning", None)
                    payload.pop("temperature", None)
                    continue
                raise RuntimeError(f"OpenRouter API error for {model_id}: {err_msg}")

            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = msg.get("content") or ""
            native_reasoning = _extract_native_reasoning(msg)
            finish_reason = (
                choice.get("finish_reason")
                or choice.get("native_finish_reason")
                or "stop"
            )
            provider_model = data.get("model") or model_id

            usage = data.get("usage") or {}
            comp_details = usage.get("completion_tokens_details") or {}
            reasoning_tokens = int(
                comp_details.get("reasoning_tokens")
                or usage.get("reasoning_tokens")
                or 0
            )
            cost_usd = float(
                usage.get("cost")
                or usage.get("total_cost")
                or 0.0
            )
            excerpt_src = content.strip() or (native_reasoning or "").strip()
            raw_excerpt = excerpt_src[:500]

            return {
                "content": content,
                "native_reasoning": native_reasoning,
                "latency_ms": latency_ms,
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "reasoning_tokens": reasoning_tokens,
                "cost_usd": cost_usd,
                "provider_model": provider_model,
                "finish_reason": str(finish_reason),
                "raw_response_excerpt": raw_excerpt,
            }

        except urllib.error.HTTPError as exc:
            last_err = exc
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            if exc.code == 400 and attempt < 2:
                # Fallback if model/provider rejected reasoning or temperature parameters
                if (
                    "reasoning" in payload
                    or "include_reasoning" in payload
                    or "temperature" in payload
                ):
                    payload.pop("reasoning", None)
                    payload.pop("include_reasoning", None)
                    payload.pop("temperature", None)
                    continue
            if exc.code in (400, 401, 403):
                raise RuntimeError(
                    f"Non-retryable HTTP {exc.code} from OpenRouter for {model_id}: {err_body}"
                ) from exc
            if attempt < 3:
                time.sleep(1.5 * (2**attempt))

        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_err = exc
            if attempt < 3:
                time.sleep(1.5 * (2**attempt))

    raise RuntimeError(
        f"OpenRouter request failed after 4 attempts for {model_id}: {last_err}"
    )
