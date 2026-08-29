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
import shlex
import subprocess
import sys
import urllib.parse
import urllib.request
import urllib.error
import urllib.parse


class LLMError(RuntimeError):
    pass


def _post_json(url, payload, api_key, timeout):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return json.loads(res.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:2000]
        # The reply body is the fastest way to see why a local server said no,
        # so it goes to the caller — but only for loopback. base_url arrives in
        # a request body, and echoing a remote reply turns this route into a
        # read primitive against anything the ComfyUI host can reach. For those,
        # the status line goes back and the body stays in the console.
        if is_local_target(url):
            raise LLMError(f"LLM server returned HTTP {e.code}: {body}") from e
        print(f"[h3_prompt_maker] HTTP {e.code} from {url}: {body}", flush=True)
        raise LLMError(
            f"LLM server returned HTTP {e.code}. 원격 주소라 응답 본문은 ComfyUI 콘솔에만 남깁니다."
        ) from e
    except urllib.error.URLError as e:
        raise LLMError(f"Cannot reach LLM server at {url}: {e.reason}") from e


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
#: "5m"    stay for five idle minutes, then unload
#: "now"   unload as soon as the answer is returned
UNLOAD_MODES = ["keep", "5m", "now"]

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


def call_openai_compatible(base_url, model, api_key, system_prompt, user_text,
                           images_base64=None, temperature=0.7, seed=-1, timeout=600,
                           max_tokens=DEFAULT_MAX_TOKENS, thinking="auto", unload_after="keep",
                           audios_base64=None, backend_name="openai_compat"):
    url = base_url.rstrip("/") + "/chat/completions"
    gemini = is_gemini_target(base_url)
    if not gemini:
        # /no_think is a Qwen convention; on Gemini it would just land in the
        # prompt as literal text. Gemini gets reasoning_effort below instead.
        user_text = apply_thinking(user_text, thinking)

    if images_base64 or audios_base64:
        content = [{"type": "text", "text": user_text}]
        for b64 in (images_base64 or []):
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{image_mime(b64)};base64,{b64}"}})
        # Only an omni model (Qwen2-Audio, Qwen2.5/3-Omni, gpt-4o-audio) has an
        # audio tower. A text or vision-only model rejects this, and the retry
        # below drops it — which is the honest outcome, not a silent success.
        for b64 in (audios_base64 or []):
            content.append({"type": "input_audio",
                            "input_audio": {"data": b64, "format": audio_format(b64)}})
    else:
        content = user_text

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
            # Google's compatibility layer speaks reasoning_effort. "none"
            # disables thinking where the model allows it (flash); a model that
            # cannot (2.5-pro) rejects it with a 400 and the retry below sheds
            # the field, which lands on the model's own default — same contract
            # as chat_template_kwargs on servers that do not know it.
            payload["reasoning_effort"] = "none" if thinking == "off" else "high"
        else:
            # vLLM and recent LM Studio builds switch Qwen3's chat template with
            # this. It is the reliable half of the switch where it is supported;
            # the /no_think token already in user_text covers everywhere else.
            payload["chat_template_kwargs"] = {"enable_thinking": thinking == "on"}

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
            msg = str(exc).lower()
            if gemini and "max_tokens" in payload and "max_tokens" in msg:
                # The 60000 default is headroom for local reasoning models, but
                # it sits above some Gemini models' output cap and Google
                # rejects it by name. Dropping the field falls back to that
                # model's own maximum — which is what the headroom meant.
                del payload["max_tokens"]
                continue
            optional = [k for k in ("chat_template_kwargs", "ttl", "keep_alive",
                                    "reasoning_effort") if k in payload]
            optional_rejected = ("400" in msg or "template" in msg or "extra" in msg
                                 or "unrecognized" in msg or "unknown" in msg)
            if optional and optional_rejected:
                for key in optional:
                    payload.pop(key, None)
                continue

            content_now = payload["messages"][1]["content"]
            parts = content_now if isinstance(content_now, list) else []
            media_rejected = ("image" in msg or "vision" in msg or "audio" in msg
                              or "content" in msg or "400" in msg)
            if media_rejected and any(p.get("type") == "input_audio" for p in parts):
                payload["messages"][1]["content"] = [
                    part for part in parts if part.get("type") != "input_audio"
                ]
                continue
            if media_rejected and any(p.get("type") == "image_url" for p in parts):
                payload["messages"][1]["content"] = user_text
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
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


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
        # Google's list endpoint says "models/gemini-2.5-flash" while the chat
        # endpoint wants the bare id; and unlike a local server, an empty model
        # here would 404 rather than mean "whatever is loaded".
        mdl = (mdl or "").strip()
        if mdl.startswith("models/"):
            mdl = mdl[len("models/"):]
        mdl = mdl or DEFAULT_GEMINI_MODEL
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
             server_model=AUTO_MODEL, max_tokens=DEFAULT_MAX_TOKENS, thinking="auto",
             unload_after="keep", audios_base64=None):
    backend = normalize_backend(backend)
    kind, url, mdl, cmd = resolve_backend(backend, base_url, model, cli_command, server_model)
    if kind == "cli":
        # A CLI runner has no template switch; the text token is all there is.
        return call_cli(cmd, system_prompt, apply_thinking(user_text, thinking), timeout=timeout)
    api_key = resolve_api_key(api_key, url)
    return call_openai_compatible(url, mdl, api_key, system_prompt, user_text,
                                  images_base64=images_base64, temperature=temperature,
                                   seed=seed, timeout=timeout, max_tokens=max_tokens,
                                   thinking=thinking, unload_after=unload_after,
                                   audios_base64=audios_base64, backend_name=backend)


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
    if normalize_backend(backend) == "lmstudio":
        models = _lmstudio_chat_models(url, key, timeout, models)
    elif is_gemini_target(url):
        models = filter_gemini_chat_models(models)
    return {"ok": True, "kind": "http", "models": models, "target": url,
            "detail": f"모델 {len(models)}개 확인됨"}


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
