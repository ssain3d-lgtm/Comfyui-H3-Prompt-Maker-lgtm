"""
LLM backends for the H3 Prompt Maker nodes.

Two families:
- "openai_compatible": any server speaking the OpenAI chat-completions API
  (LM Studio, llama.cpp llama-server, Ollama /v1, vLLM, KoboldCpp,
   OpenRouter, OpenAI, ...). The `gemini` preset points this family at
   Google's own endpoint (generativelanguage.googleapis.com) — the same
   Google models the H3 web app used — with the address, default model,
   thinking mapping and GEMINI_API_KEY handling filled in.
- "cli": any command-line model runner. The full prompt is piped to the
  command's stdin and stdout is taken as the answer
  (Claude Code: `claude -p --output-format text`, Gemini CLI: `gemini -p`, ...)
"""

import base64
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import urllib.error


class LLMError(RuntimeError):
    pass


class LLMCancelled(LLMError):
    """The caller stopped an in-flight streamed completion."""


class StreamCancel:
    """Cross-thread cancellation that also closes a blocking urllib response.

    ``threading.Event`` alone is only observed after the next SSE line arrives.
    During a long model load/prefill that can take minutes, so the aiohttp route
    attaches the active response here and can close its socket immediately when
    the browser aborts the fetch.
    """

    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._response = None

    def is_set(self):
        return self._event.is_set()

    def attach(self, response):
        with self._lock:
            if self._event.is_set():
                try:
                    response.close()
                finally:
                    raise LLMCancelled("Generation cancelled.")
            self._response = response

    def detach(self, response):
        with self._lock:
            if self._response is response:
                self._response = None

    def cancel(self):
        self._event.set()
        with self._lock:
            response = self._response
            self._response = None
        if response is not None:
            try:
                response.close()
            except Exception:
                pass


def _open_json_response(url, payload, api_key, timeout):
    """Open a JSON POST and preserve the existing safe error reporting rules."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers=headers, method="POST")
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:2000]
        # The reply body is the fastest way to see why a server said no, so it
        # goes to the caller — but only for loopback and for Google's pinned
        # host. base_url arrives in a request body, and echoing an arbitrary
        # remote reply turns this route into a read primitive against anything
        # the ComfyUI host can reach. Google's API is not that: its error body
        # is a JSON status (retired model, quota, bad key) and is exactly what
        # the person at the overlay needs. Everything else: status line back,
        # body to the console.
        if is_local_target(url):
            raise LLMError(f"LLM server returned HTTP {e.code}: {body}") from e
        if is_gemini_target(url):
            raise LLMError(f"Gemini API returned HTTP {e.code}: {body}"
                           + gemini_error_hint(e.code, body)) from e
        print(f"[h3_prompt_maker] HTTP {e.code} from {url}: {body}", flush=True)
        raise LLMError(
            f"LLM server returned HTTP {e.code}. 원격 주소라 응답 본문은 ComfyUI 콘솔에만 남깁니다."
        ) from e
    except urllib.error.URLError as e:
        raise LLMError(f"Cannot reach LLM server at {url}: {e.reason}") from e


def _post_json(url, payload, api_key, timeout):
    with _open_json_response(url, payload, api_key, timeout) as res:
        return json.loads(res.read().decode("utf-8", errors="replace"))


#: A reasoning model spends this budget thinking before it answers. At 8192 a
#: Qwen3-class model could burn nearly all of it inside <think> and emit a
#: single line of assumptions as the "answer" — which is exactly what happened.
#: An H3 prompt is ~600 words; the headroom is for the thinking, not the output.
DEFAULT_MAX_TOKENS = 60000
MIN_MAX_TOKENS = 1024
MAX_MAX_TOKENS = 1000000


def clamp_max_tokens(value, default=DEFAULT_MAX_TOKENS):
    """Keep a hand-typed value inside what a server will accept. Never raises."""
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return default
    return max(MIN_MAX_TOKENS, min(MAX_MAX_TOKENS, n))


#: What happens to the model after a generation finishes.
#: "keep"  stay resident — fastest next run, holds the VRAM
#: "close" stay resident while the overlay is open, then explicitly unload
#: "5m"    stay for five idle minutes, then unload
#: "now"   unload as soon as the answer is returned
UNLOAD_MODES = ["keep", "close", "5m", "now"]

#: Idle seconds per mode. LM Studio uses ttl for JIT-loaded models and its native
#: v1 endpoint for a guaranteed immediate unload; Ollama uses keep_alive on the
#: inference request. llama.cpp and vLLM hold one model for the process lifetime.
_TTL_SECONDS = {"5m": 300, "now": 1}


def unload_payload(mode, backend="openai_compat"):
    """Backend-specific keep-alive fields for ``mode``.

    Sending both dialects at once looks portable, but strict servers reject the
    field they do not know.  The old retry then removed *both* fields, so the
    generation succeeded while the model silently stayed in VRAM.
    """
    if mode not in _TTL_SECONDS:
        return {}
    seconds = _TTL_SECONDS[mode]
    backend = normalize_backend(backend)
    if backend == "lmstudio":
        # TTL remains a compatibility fallback for pre-v0.4 LM Studio and for
        # JIT-loaded models.  Immediate unload is also enforced through the
        # native /api/v1/models/unload endpoint after a successful generation.
        return {"ttl": seconds}
    if backend == "ollama":
        return {"keep_alive": 0 if mode == "now" else f"{seconds // 60}m"}
    return {}


#: Whether to let a reasoning model think before answering.
#: "auto"  send nothing — the model's own default
#: "off"   suppress it: the whole budget goes to the answer
#: "on"    force it on
THINKING_MODES = ["auto", "off", "on"]
DEFAULT_THINKING = "off"

#: Qwen3 and its derivatives read these as commands from the user turn. They are
#: plain text, so a model that does not know them simply ignores them — which is
#: why this is the portable half of the switch.
_THINK_TOKENS = {"off": "/no_think", "on": "/think"}


def apply_thinking(user_text, mode):
    """Append the control token for `mode`. Returns the text unchanged for auto."""
    token = _THINK_TOKENS.get(mode)
    if not token:
        return user_text
    return f"{user_text}\n\n{token}"


def _extract_text(data):
    """The answer, with a separately-returned reasoning block folded back in."""
    try:
        message = data["choices"][0]["message"]
        text = message.get("content")
    except (KeyError, IndexError, TypeError, AttributeError):
        raise LLMError(f"Unexpected LLM response shape: {str(data)[:500]}")

    # LM Studio and others hand a reasoning model's thinking back in its own
    # field instead of inline. When the model spent that field on the actual
    # prompt and left only a note in content, the answer is still here — put it
    # back as a <think> block so the parsers can recover it.
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning and isinstance(reasoning, str) and reasoning.strip():
        text = f"<think>{reasoning}</think>\n{text or ''}"
    if not text or not text.strip():
        raise LLMError("LLM returned an empty response.")
    return text


# Base64 of a file's first bytes always starts the same way, so the type can be
# read off the payload without carrying the data-URL prefix around. Labelling a
# JPEG contact sheet as image/png works on servers that sniff the bytes and
# fails on the strict ones, which is the worst kind of bug to chase.
_MAGIC = (("/9j/", "image/jpeg"), ("iVBORw0KGgo", "image/png"),
          ("R0lGOD", "image/gif"), ("UklGR", "image/webp"))


def image_mime(b64):
    for prefix, mime in _MAGIC:
        if isinstance(b64, str) and b64.startswith(prefix):
            return mime
    return "image/png"


# The chat schema carries audio as a bare base64 payload plus a format name, so
# the container has to be read off the bytes. Guessing wrong is not harmless:
# a server handed format "wav" for an mp3 decodes noise and describes it.
def audio_format(b64):
    try:
        chunk = b64[: (len(b64) // 4) * 4][:32]
        head = base64.b64decode(chunk)
    except Exception:  # noqa: BLE001 — an undecodable payload is not worth a crash
        return "wav"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:4] == b"fLaC":
        return "flac"
    if head[4:8] == b"ftyp":
        return "m4a"
    if head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return "mp3"
    return "wav"


def _chat_payload(base_url, model, system_prompt, user_text, images_base64,
                  audios_base64, temperature, seed, max_tokens, thinking,
                  unload_after, backend_name, stream=False):
    """Build one OpenAI-compatible request and return its controlled user text."""
    gemini = is_gemini_target(base_url)
    controlled_text = user_text if gemini else apply_thinking(user_text, thinking)

    if images_base64 or audios_base64:
        content = [{"type": "text", "text": controlled_text}]
        for b64 in (images_base64 or []):
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{image_mime(b64)};base64,{b64}"}})
        for b64 in (audios_base64 or []):
            content.append({"type": "input_audio",
                            "input_audio": {"data": b64, "format": audio_format(b64)}})
    else:
        content = controlled_text

    payload = {
        "model": model or "local-model",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None and seed >= 0:
        payload["seed"] = int(seed)
    payload.update(unload_payload(unload_after, backend_name))
    if thinking in ("off", "on"):
        if gemini:
            # Google's compatibility layer speaks reasoning_effort. Current
            # (3.x) models cannot switch thinking off at all and answer
            # "none" with a 400, which cost every generation a retry now that
            # off is the default; "low" is the least thinking every model
            # accepts, "high" the most. A model that still rejects the field
            # is retried without it by _retry_chat_payload.
            payload["reasoning_effort"] = "low" if thinking == "off" else "high"
        else:
            payload["chat_template_kwargs"] = {"enable_thinking": thinking == "on"}
    if stream:
        payload["stream"] = True
        # LM Studio and current OpenAI-compatible servers return exact token
        # counts in the final chunk when asked. Older servers reject this field;
        # the normal optional-capability retry below removes it automatically.
        payload["stream_options"] = {"include_usage": True}
    return payload, controlled_text, gemini


def _retry_chat_payload(payload, message, gemini, controlled_text):
    """Remove one unsupported capability and say whether the request can retry."""
    msg = message.lower()
    if gemini and "max_tokens" in payload and (
            "max_tokens" in msg or "max_output_tokens" in msg or "output token" in msg):
        # The 60000 default is headroom for local reasoning models, but it
        # sits above some Gemini models' output cap and Google rejects it by
        # its own field name (max_output_tokens). Dropping the field falls
        # back to that model's maximum — which is what the headroom meant.
        del payload["max_tokens"]
        return True

    optional = [k for k in ("stream_options", "chat_template_kwargs", "ttl", "keep_alive",
                            "reasoning_effort") if k in payload]
    optional_rejected = ("400" in msg or "template" in msg or "extra" in msg
                         or "unrecognized" in msg or "unknown" in msg)
    if optional and optional_rejected:
        for key in optional:
            payload.pop(key, None)
        return True

    content_now = payload["messages"][1]["content"]
    parts = content_now if isinstance(content_now, list) else []
    media_rejected = ("image" in msg or "vision" in msg or "audio" in msg
                      or "content" in msg or "400" in msg)
    if media_rejected and any(p.get("type") == "input_audio" for p in parts):
        payload["messages"][1]["content"] = [
            part for part in parts if part.get("type") != "input_audio"
        ]
        return True
    if media_rejected and any(p.get("type") == "image_url" for p in parts):
        payload["messages"][1]["content"] = controlled_text
        return True
    return False


def call_openai_compatible(base_url, model, api_key, system_prompt, user_text,
                           images_base64=None, temperature=0.7, seed=-1, timeout=600,
                           max_tokens=DEFAULT_MAX_TOKENS, thinking=DEFAULT_THINKING, unload_after="keep",
                           audios_base64=None, backend_name="openai_compat"):
    url = base_url.rstrip("/") + "/chat/completions"
    payload, controlled_text, gemini = _chat_payload(
        base_url, model, system_prompt, user_text, images_base64, audios_base64,
        temperature, seed, max_tokens, thinking, unload_after, backend_name)

    def post():
        return _post_json(url, payload, api_key, timeout)

    # Each retry removes one optional capability and then comes back through
    # the same error handler.  The previous nested returns stopped after an
    # unsupported-audio 400 whenever ttl/keep_alive was also present: retry 1
    # removed the TTL fields, retry 2 hit the same audio 400 and escaped before
    # the audio-shedding branch could run.
    while True:
        try:
            answer = _extract_text(post())
            break
        except LLMError as exc:
            if _retry_chat_payload(payload, str(exc), gemini, controlled_text):
                continue
            raise

    # LM Studio documents ttl as applying to JIT-loaded instances.  A model
    # loaded manually or with `lms load` can therefore ignore ttl entirely.
    # v0.4's native endpoint unloads the actual loaded instance regardless of
    # how it entered memory, which is what "즉시 언로드" promises.
    if normalize_backend(backend_name) == "lmstudio" and unload_after == "now":
        result = _unload_lmstudio_model(base_url, api_key, model, min(timeout, 15.0))
        if not result["ok"]:
            print(f"H3 Prompt Maker: LM Studio unload warning: {result['detail']}", file=sys.stderr)
    return answer


def _metric_number(mapping, *names):
    for name in names:
        value = mapping.get(name) if isinstance(mapping, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def _stream_metrics(started, first_token_at, usage, stats, answer):
    """Normalize OpenAI/LM Studio counters into the fields the overlay shows."""
    ended = time.perf_counter()
    total_ms = max(0.0, (ended - started) * 1000)
    ttft_ms = ((first_token_at - started) * 1000) if first_token_at else None
    prompt_tokens = _metric_number(usage, "prompt_tokens", "input_tokens")
    completion_tokens = _metric_number(usage, "completion_tokens", "output_tokens")
    if completion_tokens is None:
        completion_tokens = _metric_number(stats, "predicted_tokens", "completion_tokens", "output_tokens")
    if prompt_tokens is None:
        prompt_tokens = _metric_number(stats, "prompt_tokens", "input_tokens")
    reported_tps = _metric_number(stats, "tokens_per_second", "generation_tokens_per_second",
                                  "predicted_tokens_per_second")
    decode_ms = total_ms - (ttft_ms or 0.0)
    calculated_tps = ((float(completion_tokens) / (decode_ms / 1000))
                      if completion_tokens is not None and decode_ms > 0 else None)
    details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
    prompt_details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    return {
        "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
        "total_ms": round(total_ms, 1),
        "prompt_tokens": int(prompt_tokens) if prompt_tokens is not None else None,
        "completion_tokens": int(completion_tokens) if completion_tokens is not None else None,
        "reasoning_tokens": (int(_metric_number(details, "reasoning_tokens"))
                             if _metric_number(details, "reasoning_tokens") is not None else None),
        "cached_tokens": (int(_metric_number(prompt_details, "cached_tokens"))
                          if _metric_number(prompt_details, "cached_tokens") is not None else None),
        "tokens_per_second": round(float(reported_tps or calculated_tps), 2)
        if (reported_tps or calculated_tps) is not None else None,
        "output_chars": len(answer),
        "streamed": True,
    }


def _read_openai_stream(response, on_delta, cancel, started):
    content_chunks, reasoning_chunks, raw_chunks = [], [], []
    usage, stats = {}, {}
    first_token_at = None
    saw_sse = False
    try:
        for raw_line in response:
            if cancel is not None and cancel.is_set():
                raise LLMCancelled("Generation cancelled.")
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if not line.startswith("data:"):
                raw_chunks.append(raw_line)
                continue
            saw_sse = True
            data_text = line[5:].strip()
            if data_text == "[DONE]":
                break
            try:
                data = json.loads(data_text)
            except ValueError:
                continue
            if isinstance(data.get("usage"), dict):
                usage.update(data["usage"])
            if isinstance(data.get("stats"), dict):
                stats.update(data["stats"])
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0] if isinstance(choices[0], dict) else {}
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            # A few compatible servers send a complete message in the last
            # chunk rather than deltas. Treat it as one final delta.
            if not delta and isinstance(choice.get("message"), dict):
                delta = choice["message"]
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            content = delta.get("content")
            if isinstance(reasoning, str) and reasoning:
                reasoning_chunks.append(reasoning)
                first_token_at = first_token_at or time.perf_counter()
            if isinstance(content, str) and content:
                content_chunks.append(content)
                first_token_at = first_token_at or time.perf_counter()
                if on_delta:
                    on_delta(content)
    except LLMCancelled:
        raise
    except Exception as exc:
        if cancel is not None and cancel.is_set():
            raise LLMCancelled("Generation cancelled.") from exc
        raise LLMError(f"Streaming response failed: {exc}") from exc

    if cancel is not None and cancel.is_set():
        raise LLMCancelled("Generation cancelled.")

    if not saw_sse:
        try:
            data = json.loads(b"".join(raw_chunks).decode("utf-8", errors="replace"))
        except Exception as exc:
            raise LLMError(f"Unexpected streaming response: {exc}") from exc
        answer = _extract_text(data)
        if on_delta:
            on_delta(answer)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        stats = data.get("stats") if isinstance(data.get("stats"), dict) else {}
        first_token_at = first_token_at or time.perf_counter()
    else:
        content = "".join(content_chunks)
        reasoning = "".join(reasoning_chunks)
        answer = (f"<think>{reasoning}</think>\n{content}" if reasoning else content)
        if not answer.strip():
            raise LLMError("LLM returned an empty streaming response.")

    return answer, _stream_metrics(started, first_token_at, usage, stats, answer)


def call_openai_compatible_stream(base_url, model, api_key, system_prompt, user_text,
                                  images_base64=None, temperature=0.7, seed=-1, timeout=600,
                                  max_tokens=DEFAULT_MAX_TOKENS, thinking=DEFAULT_THINKING,
                                  unload_after="keep", audios_base64=None,
                                  backend_name="openai_compat", on_delta=None, cancel=None):
    """Stream one completion, returning ``(text, metrics)`` with real cancellation."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload, controlled_text, gemini = _chat_payload(
        base_url, model, system_prompt, user_text, images_base64, audios_base64,
        temperature, seed, max_tokens, thinking, unload_after, backend_name, stream=True)
    started = time.perf_counter()

    while True:
        if cancel is not None and cancel.is_set():
            raise LLMCancelled("Generation cancelled.")
        response = None
        try:
            response = _open_json_response(url, payload, api_key, timeout)
            if cancel is not None:
                cancel.attach(response)
            answer, metrics = _read_openai_stream(response, on_delta, cancel, started)
            break
        except LLMCancelled:
            raise
        except LLMError as exc:
            msg = str(exc).lower()
            # Preserve media when an older server rejects streaming. A generic
            # HTTP 400 is also used for unsupported image parts, so letting the
            # media fallback run first would silently turn a VLM request into a
            # text-only retry merely because `stream` was the unknown field.
            if "stream_options" in payload and ("stream_options" in msg or "stream options" in msg):
                payload.pop("stream_options", None)
                continue
            # Very old compatibility servers can reject streaming itself. Keep
            # generation working, but mark it non-streamed and deliver at once.
            if ("stream" in payload and "stream" in msg
                    and ("400" in msg or "unknown" in msg or "unsupported" in msg)):
                payload.pop("stream", None)
                payload.pop("stream_options", None)
                fallback_started = time.perf_counter()
                try:
                    response = _open_json_response(url, payload, api_key, timeout)
                    if cancel is not None:
                        cancel.attach(response)
                    data = json.loads(response.read().decode("utf-8", errors="replace"))
                except Exception as fallback_exc:
                    if cancel is not None and cancel.is_set():
                        raise LLMCancelled("Generation cancelled.") from fallback_exc
                    raise
                answer = _extract_text(data)
                if on_delta:
                    on_delta(answer)
                elapsed = (time.perf_counter() - fallback_started) * 1000
                metrics = {"ttft_ms": round(elapsed, 1), "total_ms": round(elapsed, 1),
                           "prompt_tokens": None, "completion_tokens": None,
                           "reasoning_tokens": None, "cached_tokens": None,
                           "tokens_per_second": None, "output_chars": len(answer),
                           "streamed": False}
                break
            if _retry_chat_payload(payload, str(exc), gemini, controlled_text):
                continue
            raise
        finally:
            if response is not None:
                if cancel is not None:
                    cancel.detach(response)
                try:
                    response.close()
                except Exception:
                    pass

    if normalize_backend(backend_name) == "lmstudio" and unload_after == "now":
        result = _unload_lmstudio_model(base_url, api_key, model, min(timeout, 15.0))
        if not result["ok"]:
            print(f"H3 Prompt Maker: LM Studio unload warning: {result['detail']}", file=sys.stderr)
    return answer, metrics


def call_cli(cli_command, system_prompt, user_text, timeout=600):
    """Pipe the whole prompt to a CLI model runner's stdin and read stdout.

    Works with e.g.:
      claude -p --output-format text
      gemini -p
      codex exec
    Note: `llama-cli` one-shot reloads the model on every call — prefer
    running `llama-server` and the openai_compatible backend instead.
    """
    full_prompt = f"{system_prompt}\n\n=== USER REQUEST ===\n{user_text}"
    # No shell. A workflow JSON (or the PNG someone shares) carries cli_command as
    # data, so `shell=True` would make "curl ... | sh" run on Queue.
    try:
        argv = shlex.split(cli_command)
    except ValueError as e:
        raise LLMError(f"Could not parse cli_command: {e}")
    if not argv:
        raise LLMError("cli_command is empty.")
    try:
        proc = subprocess.run(
            argv,
            shell=False,
            input=full_prompt.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise LLMError(f"CLI command timed out after {timeout}s: {cli_command}")
    except OSError as e:
        raise LLMError(f"Failed to run CLI command '{cli_command}': {e}")

    stdout = proc.stdout.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0 and not stdout:
        stderr = proc.stderr.decode("utf-8", errors="replace")[:2000]
        raise LLMError(f"CLI command failed (exit {proc.returncode}): {stderr}")
    if not stdout:
        raise LLMError("CLI command produced no output.")
    return stdout


AUTO_MODEL = "(auto)"

# Google's Gemini API through its OpenAI-compatible endpoint — the same Google
# models the H3 web app called, spoken in the chat dialect this file already
# knows. Host is pinned here: key routing below trusts it by name.
GEMINI_HOST = "generativelanguage.googleapis.com"
GEMINI_BASE_URL = f"https://{GEMINI_HOST}/v1beta/openai"
# Google's own model list. Unlike the OpenAI-shaped one it says what each
# model can do (supportedGenerationMethods), so chat models are picked by
# capability rather than by guessing from the name.
GEMINI_NATIVE_MODELS_URL = f"https://{GEMINI_HOST}/v1beta/models"
# Last resort only, when the live list cannot be read. Google retires a Flash
# generation roughly yearly (2.5 Flash went dark on 2026-06-17 and took the
# previous hard-coded default with it), so the list is authoritative and this
# is the fallback: the newest stable Flash at the time of writing, free tier
# 10 RPM / 1,500 requests a day.
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"


def is_gemini_target(url):
    """True when `url` points at Google's Gemini API host."""
    try:
        return (urllib.parse.urlsplit(url or "").hostname or "").lower() == GEMINI_HOST
    except Exception:  # noqa: BLE001 — an unparseable URL is not Gemini
        return False


# Backend presets — pick one and the standard address/command fills itself in.
# (Same approach as ComfyUI-LLM-Hub: aliases share the OpenAI-compatible client,
#  they only differ in the default port.)
# Only these loopback servers are probed by discover_local_models. The gemini
# preset stays out: it is remote and keyed, so probing it on every page load
# would stall the UI and spend quota for nothing.
LOCAL_PRESET_BASE_URLS = {
    "lmstudio": "http://127.0.0.1:1234/v1",
    "ollama": "http://127.0.0.1:11434/v1",
    "llamacpp": "http://127.0.0.1:8080/v1",
    "vllm": "http://127.0.0.1:8000/v1",
}
PRESET_BASE_URLS = {**LOCAL_PRESET_BASE_URLS, "gemini": GEMINI_BASE_URL}
PRESET_CLI_COMMANDS = {
    "claude_cli": "claude -p --output-format text",
    "gemini_cli": "gemini -p",
    "codex_cli": "codex exec",
}
HTTP_BACKENDS = ["lmstudio", "ollama", "llamacpp", "vllm", "gemini", "openai_compat"]
CLI_BACKEND_LIST = ["claude_cli", "gemini_cli", "codex_cli", "custom_cli"]
BACKEND_NAMES = HTTP_BACKENDS + CLI_BACKEND_LIST

# v1.0 saved workflows used these names
_BACKEND_ALIASES = {"openai_compatible": "openai_compat", "cli": "custom_cli"}


def normalize_backend(name):
    return _BACKEND_ALIASES.get(name, name)


_MODEL_CACHE = {"at": 0.0, "ids": []}
_MODEL_CACHE_TTL = 30.0


_NON_CHAT_ARCHITECTURES = {"clip", "nomic-bert"}
_CHAT_MODEL_TYPES = {"llm", "vlm"}


def _model_ids(payload):
    """Unique ids from an OpenAI-compatible ``GET /v1/models`` response."""
    ids = []
    for model in (payload.get("data") or []):
        mid = model.get("id") if isinstance(model, dict) else None
        if mid and mid not in ids:
            ids.append(mid)
    return ids


# Everything Google's model list exposes that cannot hold a chat conversation:
# embedders, image/video generators, TTS voices, the Live-API duplex models and
# the native-audio variants. Offering one of these as the prompt model fails at
# generate time with a shape error — the worst place to learn it.
_GEMINI_NON_CHAT_MARKERS = ("embedding", "imagen", "image", "veo", "tts",
                            "live", "audio", "aqa")


def filter_gemini_chat_models(ids):
    """Chat-capable Gemini/Gemma ids from Google's list, bare (no models/ prefix)."""
    out = []
    for mid in ids:
        if not isinstance(mid, str):
            continue
        base = mid[len("models/"):] if mid.startswith("models/") else mid
        low = base.lower()
        if not low.startswith(("gemini", "gemma")):
            continue
        if any(marker in low for marker in _GEMINI_NON_CHAT_MARKERS):
            continue
        if base and base not in out:
            out.append(base)
    return out


def _gemini_native_models(api_key, timeout):
    """Every model Google lists for this key that answers generateContent.

    Follows nextPageToken: the list is past fifty entries and the newest
    models sit wherever Google puts them, so a single page is not enough.
    """
    names, token, pages = [], "", 0
    while pages < 10:
        query = "?pageSize=1000" + (f"&pageToken={urllib.parse.quote(token)}" if token else "")
        req = urllib.request.Request(GEMINI_NATIVE_MODELS_URL + query,
                                     headers={"x-goog-api-key": api_key})
        with urllib.request.urlopen(req, timeout=timeout) as res:
            payload = json.loads(res.read().decode("utf-8", errors="replace"))
        rows = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise LLMError(f"Unexpected model list shape: {str(payload)[:200]}")
        for row in rows:
            if not isinstance(row, dict):
                continue
            methods = row.get("supportedGenerationMethods") or []
            if "generateContent" in methods and isinstance(row.get("name"), str):
                names.append(row["name"])
        token = payload.get("nextPageToken") or ""
        pages += 1
        if not token:
            break
    return names


def gemini_chat_models(api_key, timeout=6.0, fallback_ids=None):
    """Chat-capable Gemini ids for this key, newest source first.

    The native list is authoritative (capability flags, all pages). If it
    cannot be read the OpenAI-shaped ids already fetched by the probe are
    filtered by name instead, so the dialog still gets a list.
    """
    try:
        return filter_gemini_chat_models(_gemini_native_models(api_key, timeout))
    except Exception as exc:  # noqa: BLE001 — a degraded list beats no list
        print(f"[h3_prompt_maker] Gemini native model list unavailable ({exc}); "
              f"falling back to the OpenAI-compatible list", flush=True)
        return filter_gemini_chat_models(fallback_ids or [])


# gemini-3.6-flash / gemini-3-flash-preview / gemini-3.8-flash-preview-05-20.
# A preview suffix may only be a date or build number: letters after -preview
# name a different product (-preview-image-generation), as do -lite and -tts.
_FLASH_RE = re.compile(r"^gemini-(\d+(?:\.\d+)*)-flash(-preview(?:-[\d.-]+)?)?$")


def _version_key(text):
    return tuple(int(part) for part in text.split("."))


def pick_default_gemini_model(ids):
    """The newest stable `gemini-N-flash`, a preview only when no stable exists.

    Flash is the free-tier workhorse (1,500 requests a day on 3.6) and the
    closest match to what the web app ran on; Pro models have no free quota.
    Picking from the live list is what keeps (auto) working when Google
    retires a generation — which is exactly how the old fixed default died.
    """
    stable, preview = [], []
    for mid in ids or []:
        m = _FLASH_RE.match(mid) if isinstance(mid, str) else None
        if not m:
            continue
        (preview if m.group(2) else stable).append((_version_key(m.group(1)), mid))
    for pool in (stable, preview):
        if pool:
            return max(pool)[1]
    return DEFAULT_GEMINI_MODEL


_GEMINI_DEFAULT_CACHE = {"at": 0.0, "key": "", "id": ""}
_GEMINI_DEFAULT_TTL = 600.0


def default_gemini_model(api_key, timeout=6.0):
    """What (auto) means for Gemini right now, cached briefly per key."""
    now = time.time()
    cache = _GEMINI_DEFAULT_CACHE
    if cache["id"] and cache["key"] == api_key and now - cache["at"] < _GEMINI_DEFAULT_TTL:
        return cache["id"]
    ids = gemini_chat_models(api_key, timeout)
    pick = pick_default_gemini_model(ids)
    if ids:
        cache.update({"at": now, "key": api_key, "id": pick})
    return pick


def gemini_error_hint(code, body):
    """One line telling the person at the overlay what to do about Google's answer."""
    low = (body or "").lower()
    if code == 404 or "not_found" in low:
        return ("\n→ 이 모델 ID는 더 이상 없습니다. 구글은 구세대 모델을 순차 종료합니다 "
                "(2.5 Flash/Pro 2026-06-17, 2.5 Flash-Lite 2026-07-22). ⚙️ 모델 연결에서 "
                "🔌 연결 확인을 눌러 현재 목록에서 다시 고르거나, (auto)로 두면 최신 안정 Flash를 "
                "자동으로 씁니다.")
    if code == 429:
        if "limit: 0" in low:
            return ("\n→ 이 모델은 무료 등급이 없습니다 (limit 0). Flash 계열"
                    f"(예: {DEFAULT_GEMINI_MODEL}, 무료 일 1,500회)을 고르거나 결제를 활성화하세요.")
        if "perday" in low or "per day" in low:
            return "\n→ 오늘의 무료 일일 한도를 다 썼습니다. 내일 다시 시도하거나 결제를 활성화하세요."
        return ("\n→ 분당 한도(무료 등급 Flash 10회/분) 초과입니다. 잠시 후 다시 시도하세요. "
                "Queue에 여러 개를 한 번에 넣었다면 그게 원인입니다.")
    if code in (400, 401) and ("api key" in low or "api_key" in low):
        return ("\n→ API 키가 올바르지 않습니다. https://aistudio.google.com/apikey 에서 발급한 키를 "
                "API 키 칸에 넣거나 GEMINI_API_KEY 환경변수로 두세요.")
    if code == 403:
        return ("\n→ 키 권한 문제입니다. AI Studio에서 만든 키인지, 프로젝트에 Generative Language API가 "
                "켜져 있는지, 무료 등급 지원 지역인지 확인하세요.")
    return ""


def _filter_lmstudio_chat_models(openai_ids, metadata):
    """Drop projectors and embedding models using LM Studio's rich metadata.

    ``/v1/models`` deliberately exposes every downloaded GGUF when JIT loading
    is enabled, and its OpenAI-shaped rows do not reliably distinguish a chat
    model from an mmproj/CLIP file. LM Studio's native model list does. Keep
    the OpenAI order because those are the ids users already see and save.
    """
    rows = None
    if isinstance(metadata, dict):
        rows = metadata.get("models")
        if not isinstance(rows, list):
            rows = metadata.get("data")
    if not isinstance(rows, list):
        return list(openai_ids)

    allowed = []
    saw_metadata = False
    for model in rows:
        if not isinstance(model, dict):
            continue
        model_type = str(model.get("type") or "").lower()
        architecture = str(model.get("arch") or model.get("architecture") or "").lower()
        if model_type or architecture:
            saw_metadata = True
        if model_type and model_type not in _CHAT_MODEL_TYPES:
            continue
        if architecture in _NON_CHAT_ARCHITECTURES:
            continue
        identifiers = [model.get("id"), model.get("key"), model.get("selected_variant")]
        identifiers.extend(model.get("variants") or [])
        for mid in identifiers:
            if isinstance(mid, str) and mid and mid not in allowed:
                allowed.append(mid)

    if not saw_metadata:
        return list(openai_ids)
    allowed_set = set(allowed)
    return [mid for mid in openai_ids if mid in allowed_set]


def _lmstudio_metadata_urls(openai_base_url):
    """LM Studio native model lists, current v1 first and legacy v0 second."""
    parts = urllib.parse.urlsplit(openai_base_url.rstrip("/"))
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    base = path.rstrip("/")
    make_url = lambda suffix: urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, base + suffix, "", ""))
    return [make_url("/api/v1/models"), make_url("/api/v0/models")]


def _lmstudio_native_models(openai_base_url, api_key, timeout):
    """LM Studio v1 model metadata, or ``None`` on pre-v0.4 servers."""
    endpoint = _lmstudio_metadata_urls(openai_base_url)[0]
    req = urllib.request.Request(endpoint)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            payload = json.loads(res.read().decode("utf-8", errors="replace"))
    except Exception:
        return None
    rows = payload.get("models") if isinstance(payload, dict) else None
    return rows if isinstance(rows, list) else None


def _lmstudio_matching_rows(rows, model):
    """Rows whose key, selected variant or variant list names ``model``."""
    matches = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        identifiers = [row.get("id"), row.get("key"), row.get("selected_variant")]
        variants = row.get("variants")
        if isinstance(variants, list):
            identifiers.extend(variants)
        loaded = row.get("loaded_instances")
        if isinstance(loaded, list):
            identifiers.extend(x.get("id") for x in loaded if isinstance(x, dict))
        if model in identifiers:
            matches.append(row)
    return matches


def _load_lmstudio_model(openai_base_url, api_key, model, timeout):
    """Load through LM Studio's native v1 API; report unsupported for old builds."""
    rows = _lmstudio_native_models(openai_base_url, api_key, timeout)
    if rows is None:
        return {"ok": False, "supported": False, "detail": "native v1 model API unavailable"}
    matches = _lmstudio_matching_rows(rows, model)
    if matches and any(row.get("loaded_instances") for row in matches):
        return {"ok": True, "supported": True, "detail": f"{model} 이미 메모리에 로드됨"}
    endpoint = _lmstudio_metadata_urls(openai_base_url)[0] + "/load"
    try:
        data = _post_json(endpoint, {"model": model}, api_key, timeout)
    except Exception as exc:
        return {"ok": False, "supported": True, "detail": f"LM Studio native load failed: {exc}"}
    instance = data.get("instance_id") if isinstance(data, dict) else None
    return {"ok": True, "supported": True,
            "detail": f"{model} 메모리에 로드됨" + (f" ({instance})" if instance else "")}


def _unload_lmstudio_model(openai_base_url, api_key, model, timeout):
    """Unload every loaded instance of the selected LM Studio model."""
    rows = _lmstudio_native_models(openai_base_url, api_key, timeout)
    if rows is None:
        return {"ok": False, "supported": False,
                "detail": "native v1 model API unavailable; ttl fallback was used"}
    matches = _lmstudio_matching_rows(rows, model)
    instance_ids = []
    for row in matches:
        for instance in row.get("loaded_instances") or []:
            iid = instance.get("id") if isinstance(instance, dict) else None
            if iid and iid not in instance_ids:
                instance_ids.append(iid)
    if not instance_ids:
        return {"ok": True, "supported": True, "detail": f"{model} 이미 언로드됨"}

    endpoint = _lmstudio_metadata_urls(openai_base_url)[0] + "/unload"
    for instance_id in instance_ids:
        try:
            _post_json(endpoint, {"instance_id": instance_id}, api_key, timeout)
        except Exception as exc:
            return {"ok": False, "supported": True,
                    "detail": f"{instance_id} native unload failed: {exc}"}
    return {"ok": True, "supported": True,
            "detail": f"{len(instance_ids)}개 인스턴스 언로드됨"}


def _lmstudio_chat_models(openai_base_url, api_key, timeout, fallback_ids):
    """Return metadata-filtered ids, falling back on older LM Studio builds."""
    for endpoint in _lmstudio_metadata_urls(openai_base_url):
        req = urllib.request.Request(endpoint)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                metadata = json.loads(res.read().decode("utf-8", errors="replace"))
        except Exception:
            continue
        rows = metadata.get("models") if isinstance(metadata, dict) else None
        if not isinstance(rows, list) and isinstance(metadata, dict):
            rows = metadata.get("data")
        if isinstance(rows, list):
            return _filter_lmstudio_chat_models(fallback_ids, metadata)
    return list(fallback_ids)


def discover_local_models(timeout=0.8):
    """Model ids from LLM servers running on THIS machine.

    Loopback standard ports only (LM Studio 1234 / Ollama 11434 /
    llama.cpp 8080 / vLLM 8000): closed local ports refuse instantly so
    probing is effectively free, while probing remote hosts would stall
    every page load. Results are cached for a short TTL. Failures are
    silent — most users run only one (or none) of these servers.
    """
    import time
    now = time.time()
    if now - _MODEL_CACHE["at"] < _MODEL_CACHE_TTL:
        return list(_MODEL_CACHE["ids"])
    ids = []
    for url in LOCAL_PRESET_BASE_URLS.values():
        try:
            req = urllib.request.Request(url.rstrip("/") + "/models")
            with urllib.request.urlopen(req, timeout=timeout) as res:
                data = json.loads(res.read().decode("utf-8", errors="replace"))
            found = _model_ids(data)
            if url == LOCAL_PRESET_BASE_URLS["lmstudio"]:
                found = _lmstudio_chat_models(url, "", timeout, found)
            ids.extend(mid for mid in found if mid not in ids)
        except Exception:
            continue
    _MODEL_CACHE["at"] = now
    _MODEL_CACHE["ids"] = ids
    return list(ids)


# The routes ride on ComfyUI's own server, which has no auth, and a workflow is
# a file people share. So neither an HTTP body nor a saved widget is a safe
# source for a command line: `shell=False` stops metacharacter chaining, but it
# cannot stop `sh -c "..."`, which names the interpreter outright. What may run
# is therefore fixed here — a preset, or a command the machine owner exported.
CLI_COMMAND_ENV = "H3_CLI_COMMAND"


def resolve_cli_command(backend, requested=""):
    """The command that may actually run, ignoring whatever the caller asked for.

    Presets are commands the user already installed and chose by name. custom_cli
    runs only what is in $H3_CLI_COMMAND, which takes shell access to set — so a
    shared workflow, or an unauthenticated POST, cannot pick the program.
    """
    backend = normalize_backend(backend)
    preset = PRESET_CLI_COMMANDS.get(backend)
    if preset:
        return preset
    allowed = os.environ.get(CLI_COMMAND_ENV, "").strip()
    if allowed:
        return allowed
    asked = (requested or "").strip()
    raise LLMError(
        f"'{backend}'로 실행할 명령이 지정되지 않았습니다. 임의의 명령을 요청 본문에서 받아 "
        f"실행하지 않기 위해, 사용자 지정 CLI는 환경변수 {CLI_COMMAND_ENV} 에 넣은 값만 씁니다 "
        f"(ComfyUI를 시작하는 셸에서 export). 프리셋 백엔드(claude_cli / gemini_cli / codex_cli)는 "
        f"그대로 쓸 수 있습니다."
        + (f" 설정창에 적힌 '{asked[:60]}' 은(는) 무시되었습니다." if asked else "")
    )


def resolve_backend(backend, base_url, model, cli_command, server_model=AUTO_MODEL):
    """Turn the widget values into a concrete (kind, url, model, command)."""
    backend = normalize_backend(backend)
    if backend in CLI_BACKEND_LIST:
        return ("cli", "", "", resolve_cli_command(backend, cli_command))
    url = (base_url or "").strip() or PRESET_BASE_URLS.get(backend, "")
    if not url:
        raise LLMError(f"Backend '{backend}' needs base_url (an OpenAI-compatible /v1 address).")
    mdl = model
    if server_model and server_model != AUTO_MODEL:
        mdl = server_model
    if backend == "gemini" or is_gemini_target(url):
        # Google's list endpoint says "models/gemini-3.6-flash" while the chat
        # endpoint wants the bare id. An empty model stays empty here: it means
        # (auto), which call_llm resolves against the live list — a fixed name
        # in this spot is how the retired 2.5 Flash became a 404 for everyone.
        mdl = (mdl or "").strip()
        if mdl.startswith("models/"):
            mdl = mdl[len("models/"):]
    return ("http", url, mdl, "")


HOST_ALLOWLIST_ENV = "H3_LLM_ALLOWED_HOSTS"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"}


def is_local_target(url):
    """True for a loopback address. Anything else is somebody else's machine."""
    try:
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
    except Exception:  # noqa: BLE001 — an unparseable URL is not local
        return False
    return host in _LOCAL_HOSTS


def _host_allowed_for_env_key(url):
    if is_local_target(url):
        return True
    # The gemini preset's pinned host is Google's API itself — sending the
    # GEMINI key there is that key's whole purpose, so it needs no allowlist
    # entry. Only the host is trusted; the key chosen for it is Gemini's own
    # (see _env_key_vars), never OPENAI_API_KEY.
    if is_gemini_target(url):
        return True
    try:
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
    except Exception:  # noqa: BLE001
        return False
    allowed = [h.strip().lower() for h in os.environ.get(HOST_ALLOWLIST_ENV, "").split(",")]
    return bool(host) and host in [h for h in allowed if h]


def _env_key_vars(target_url):
    """Which env keys may serve `target_url`. Google's host gets Google's keys
    only — a fallback that reached OPENAI_API_KEY first would send the wrong
    vendor's key as a Bearer header and fail as a confusing 401."""
    if target_url is not None and is_gemini_target(target_url):
        return ("GEMINI_API_KEY", "GOOGLE_API_KEY", "H3_LLM_API_KEY")
    return ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "H3_LLM_API_KEY")


def resolve_api_key(api_key, target_url=None):
    """Prefer the typed key, else the environment — so a shared workflow need not carry one.

    The environment fallback is deliberately NOT unconditional. base_url arrives
    in the request body, so an unconditional fallback hands the user's real
    OPENAI_API_KEY to whatever host a caller names, as an Authorization header.
    An env key therefore travels only to loopback, to Google's pinned Gemini
    host, or to a host the machine owner listed in $H3_LLM_ALLOWED_HOSTS. A key
    typed into the dialog is the user's explicit choice for that address and
    always travels.
    """
    if (api_key or "").strip():
        return api_key.strip()
    if target_url is not None and not _host_allowed_for_env_key(target_url):
        return ""
    for var in _env_key_vars(target_url):
        v = os.environ.get(var, "").strip()
        if v:
            return v
    return ""


def call_llm(backend, base_url, model, api_key, cli_command, system_prompt,
             user_text, images_base64=None, temperature=0.7, seed=-1, timeout=600,
             server_model=AUTO_MODEL, max_tokens=DEFAULT_MAX_TOKENS, thinking=DEFAULT_THINKING,
             unload_after="keep", audios_base64=None):
    backend = normalize_backend(backend)
    kind, url, mdl, cmd = resolve_backend(backend, base_url, model, cli_command, server_model)
    if kind == "cli":
        # A CLI runner has no template switch; the text token is all there is.
        return call_cli(cmd, system_prompt, apply_thinking(user_text, thinking), timeout=timeout)
    api_key = resolve_api_key(api_key, url)
    if is_gemini_target(url) and not mdl:
        mdl = default_gemini_model(api_key, timeout=min(timeout, 8.0))
        print(f"[h3_prompt_maker] gemini (auto) → {mdl}", flush=True)
    return call_openai_compatible(url, mdl, api_key, system_prompt, user_text,
                                  images_base64=images_base64, temperature=temperature,
                                   seed=seed, timeout=timeout, max_tokens=max_tokens,
                                   thinking=thinking, unload_after=unload_after,
                                   audios_base64=audios_base64, backend_name=backend)


def stream_llm(backend, base_url, model, api_key, cli_command, system_prompt,
               user_text, images_base64=None, temperature=0.7, seed=-1, timeout=600,
               server_model=AUTO_MODEL, max_tokens=DEFAULT_MAX_TOKENS,
               thinking=DEFAULT_THINKING, unload_after="keep", audios_base64=None,
               on_delta=None, cancel=None):
    """Streaming counterpart of :func:`call_llm` used by the overlay route."""
    backend = normalize_backend(backend)
    kind, url, mdl, cmd = resolve_backend(backend, base_url, model, cli_command, server_model)
    if kind == "cli":
        if cancel is not None and cancel.is_set():
            raise LLMCancelled("Generation cancelled.")
        started = time.perf_counter()
        answer = call_cli(cmd, system_prompt, apply_thinking(user_text, thinking), timeout=timeout)
        if cancel is not None and cancel.is_set():
            raise LLMCancelled("Generation cancelled.")
        if on_delta:
            on_delta(answer)
        elapsed = (time.perf_counter() - started) * 1000
        return answer, {
            "ttft_ms": round(elapsed, 1), "total_ms": round(elapsed, 1),
            "prompt_tokens": None, "completion_tokens": None,
            "reasoning_tokens": None, "cached_tokens": None,
            "tokens_per_second": None, "output_chars": len(answer),
            "streamed": False,
        }
    key = resolve_api_key(api_key, url)
    if is_gemini_target(url) and not mdl:
        mdl = default_gemini_model(key, timeout=min(timeout, 8.0))
        print(f"[h3_prompt_maker] gemini (auto) → {mdl}", flush=True)
    return call_openai_compatible_stream(
        url, mdl, key, system_prompt, user_text,
        images_base64=images_base64, temperature=temperature, seed=seed,
        timeout=timeout, max_tokens=max_tokens, thinking=thinking,
        unload_after=unload_after, audios_base64=audios_base64,
        backend_name=backend, on_delta=on_delta, cancel=cancel)


def unload_model(backend, base_url, api_key, model, timeout=20.0):
    """Explicitly release a selected model for the overlay's close/manual action."""
    if not model:
        return {"ok": False, "supported": True, "detail": "언로드할 모델을 먼저 선택하세요."}
    try:
        _kind, url, _m, _c = resolve_backend(backend, base_url, model, "")
    except LLMError as exc:
        return {"ok": False, "supported": True, "detail": str(exc)}
    if is_gemini_target(url):
        return {"ok": True, "supported": False, "detail": "클라우드 모델은 로컬 VRAM을 사용하지 않습니다."}
    key = resolve_api_key(api_key, url)
    normalized = normalize_backend(backend)
    if normalized == "lmstudio":
        return _unload_lmstudio_model(url, key, model, timeout)
    if normalized == "ollama":
        parts = urllib.parse.urlsplit(url)
        path = parts.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        endpoint = urllib.parse.urlunsplit(
            (parts.scheme, parts.netloc, path + "/api/generate", "", ""))
        try:
            _post_json(endpoint, {"model": model, "keep_alive": 0, "stream": False}, key, timeout)
            return {"ok": True, "supported": True, "detail": f"{model} 언로드됨"}
        except Exception as exc:
            return {"ok": False, "supported": True, "detail": f"Ollama 언로드 실패: {exc}"}
    return {"ok": False, "supported": False,
            "detail": "이 백엔드는 실행 중인 서버 프로세스가 모델을 소유하므로 여기서 언로드할 수 없습니다."}


def probe_backend(backend, base_url, api_key, cli_command, timeout=6.0):
    """Answer one question: can this configuration actually be reached?

    The settings dialog used to save an address and find out at generate time,
    which is the worst moment — the failure arrives as a fallback template. This
    is the same resolution path a real call takes, so a green result here means
    the next generation reaches the same place.

    Returns {ok, kind, detail, models, target}. Never raises.
    """
    try:
        kind, url, _mdl, cmd = resolve_backend(backend, base_url, "", cli_command)
    except LLMError as exc:
        return {"ok": False, "kind": "", "detail": str(exc), "models": [], "target": ""}

    if kind == "cli":
        import shutil
        try:
            exe = shlex.split(cmd)[0]
        except ValueError as exc:
            return {"ok": False, "kind": "cli", "detail": f"명령을 해석할 수 없습니다: {exc}",
                    "models": [], "target": cmd}
        found = shutil.which(exe)
        if not found:
            return {"ok": False, "kind": "cli", "models": [], "target": cmd,
                    "detail": f"'{exe}' 실행 파일을 PATH에서 찾지 못했습니다. "
                              f"ComfyUI를 실행한 셸에서 '{exe}'가 동작하는지 확인하세요."}
        # A CLI backend has no model list to offer — the subscription decides.
        return {"ok": True, "kind": "cli", "models": [], "target": cmd,
                "detail": f"{exe} 확인됨 ({found})"}

    endpoint = url.rstrip("/") + "/models"
    req = urllib.request.Request(endpoint)
    key = resolve_api_key(api_key, url)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            payload = json.loads(res.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        hint = " API 키를 확인하세요." if exc.code in (401, 403) else ""
        if exc.code in (401, 403) and is_gemini_target(url):
            hint = (" Gemini API 키가 필요합니다 — https://aistudio.google.com/apikey 에서 "
                    "발급해 API 키 칸에 넣거나 GEMINI_API_KEY 환경변수로 두세요.")
        return {"ok": False, "kind": "http", "models": [], "target": url,
                "detail": f"HTTP {exc.code} {exc.reason}.{hint}"}
    except Exception as exc:
        return {"ok": False, "kind": "http", "models": [], "target": url,
                "detail": f"{url} 에 연결하지 못했습니다 ({type(exc).__name__}). "
                          f"서버가 켜져 있고 주소가 맞는지 확인하세요."}

    models = _model_ids(payload)
    detail = ""
    if normalize_backend(backend) == "lmstudio":
        models = _lmstudio_chat_models(url, key, timeout, models)
    elif is_gemini_target(url):
        models = gemini_chat_models(key, timeout, fallback_ids=models)
        # Say what (auto) will run, so nobody has to guess which Flash is current.
        detail = f" · (auto) = {pick_default_gemini_model(models)}"
    return {"ok": True, "kind": "http", "models": models, "target": url,
            "detail": f"모델 {len(models)}개 확인됨{detail}"}


def warm_up_model(backend, base_url, api_key, model, timeout=180.0):
    """Make the server load `model` into memory now, rather than mid-generation.

    LM Studio v0.4+ uses its native load endpoint, which works even when JIT is
    disabled. Other servers (and older LM Studio builds) use a one-token
    completion as the portable fallback.

    Returns {ok, detail}. Never raises.
    """
    if not model:
        return {"ok": False, "detail": "로드할 모델을 먼저 선택하세요."}
    try:
        _kind, url, _m, _c = resolve_backend(backend, base_url, model, "")
    except LLMError as exc:
        return {"ok": False, "detail": str(exc)}
    if is_gemini_target(url):
        # Nothing to page in on a cloud API — and the fallback below would
        # spend a real (billable) completion request just to say so.
        return {"ok": True, "detail": "Gemini는 클라우드 API라 미리 로드할 필요가 없습니다."}
    key = resolve_api_key(api_key, url)
    if normalize_backend(backend) == "lmstudio":
        chat_models = _lmstudio_chat_models(url, key, min(timeout, 6.0), [model])
        if model not in chat_models:
            return {"ok": False,
                    "detail": f"{model}은(는) 채팅 모델이 아닙니다 (mmproj/embedding 제외)."}
        native = _load_lmstudio_model(url, key, model, timeout)
        if native["ok"]:
            return {"ok": True, "detail": native["detail"]}
        # LM Studio before v0.4 has no native load endpoint. Keep the existing
        # one-token JIT fallback for those builds, but do not hide a real error
        # from a server that does advertise the native API.
        if native["supported"]:
            return {"ok": False, "detail": native["detail"]}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    try:
        _post_json(url.rstrip("/") + "/chat/completions", body, key, timeout)
    except Exception as exc:
        return {"ok": False, "detail": f"{model} 로드 실패: {exc}"}
    return {"ok": True, "detail": f"{model} 메모리에 로드됨"}
