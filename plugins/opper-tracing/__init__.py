"""Opper tracing plugin for Hermes.

Lifecycle:
- on_session_start : create a root "hermes-session" span
- pre_llm_call     : inject session span ID + trace ID as headers so the
                     executor's own span becomes the turn span
                     → hermes-session → hermes (executor) → __api_completions
- on_session_end   : close the root session span
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Root session spans keyed by session_id
_session_spans: dict[str, dict[str, Any]] = {}

# Per-tool span queues keyed by (task_id, tool_name) — FIFO to handle parallel calls
_tool_spans: dict[tuple, list] = {}


def _api_key() -> str:
    return os.getenv("OPPER_API_KEY", "").strip()


def _api_root() -> str:
    return "https://api.opper.ai"


def _is_opper() -> bool:
    if not _api_key():
        return False
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        return "opper" in resolve_runtime_provider().get("provider", "").lower()
    except Exception:
        return False


def _now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()




def _close_span(span_id: str) -> None:
    import httpx
    try:
        httpx.patch(
            f"{_api_root()}/v2/spans/{span_id}",
            json={"end_time": _now()},
            headers={"Authorization": f"Bearer {_api_key()}"},
            timeout=5.0,
        )
    except Exception as exc:
        logger.debug("opper-tracing: close_span %s failed: %s", span_id, exc)


# ---------------------------------------------------------------------------
# Hook handlers
# ---------------------------------------------------------------------------

def _on_session_start(session_id: str = "", **_: Any) -> None:
    if not _is_opper():
        return
    span = _create_span(name="hermes-session", input_text=session_id)
    if span and span.get("id"):
        _session_spans[session_id] = span
        logger.debug("opper-tracing: session span %s", span["id"])
        # Close the span when the process exits (on_session_end fires per-turn,
        # not per-process, so atexit is more accurate for session lifetime).
        import atexit
        atexit.register(_close_span, span["id"])


def _on_pre_llm_call(session_id: str = "", **_: Any) -> dict | None:
    span = _session_spans.get(session_id)
    if not span:
        return None
    headers: dict[str, str] = {"X-Opper-Parent-Span-Id": span["id"]}
    if span.get("trace_id"):
        headers["X-Opper-Trace-Id"] = span["trace_id"]
    return {"headers": headers}


def _on_pre_tool_call(
    tool_name: str = "",
    args: Any = None,
    task_id: str = "",
    **_: Any,
) -> None:
    # pre_tool_call doesn't pass session_id — use the first active session span
    session_span = next(iter(_session_spans.values()), None)
    if not session_span:
        return
    import json
    input_text = f"{tool_name}: {json.dumps(args, default=str)[:500]}" if args else tool_name
    parent_id = session_span["id"]
    span = _create_span(name=f"tool:{tool_name}", input_text=input_text, parent_id=parent_id)
    if span and span.get("id"):
        key = (task_id, tool_name)
        _tool_spans.setdefault(key, []).append(span)
        logger.debug("opper-tracing: tool span %s for %s", span["id"], tool_name)


def _create_span(name: str, input_text: str, parent_id: str = "") -> dict[str, Any] | None:
    import httpx
    body: dict[str, Any] = {"name": name, "type": "tool", "input": input_text, "start_time": _now()}
    if parent_id:
        body["parent_id"] = parent_id
    try:
        resp = httpx.post(
            f"{_api_root()}/v2/spans",
            json=body,
            headers={"Authorization": f"Bearer {_api_key()}"},
            timeout=5.0,
        )
        if resp.status_code in (200, 201):
            return resp.json().get("data", {})
    except Exception as exc:
        logger.debug("opper-tracing: create_span failed: %s", exc)
    return None


def _on_post_tool_call(
    tool_name: str = "",
    result: Any = None,
    task_id: str = "",
    **_: Any,
) -> None:
    key = (task_id, tool_name)
    queue = _tool_spans.get(key)
    if not queue:
        return
    span = queue.pop(0)
    if not queue:
        _tool_spans.pop(key, None)
    if not span or not span.get("id"):
        return
    import json
    output = str(result)[:500] if result is not None else ""
    import httpx
    try:
        httpx.patch(
            f"{_api_root()}/v2/spans/{span['id']}",
            json={"end_time": _now(), "output": output},
            headers={"Authorization": f"Bearer {_api_key()}"},
            timeout=5.0,
        )
    except Exception as exc:
        logger.debug("opper-tracing: close tool span failed: %s", exc)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(ctx: Any) -> None:
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    logger.debug("opper-tracing plugin registered")
