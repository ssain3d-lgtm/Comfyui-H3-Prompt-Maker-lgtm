"""
LLM backends for the H3 Prompt Maker nodes.

Two families:
- "openai_compatible": any server speaking the OpenAI chat-completions API
  (LM Studio, llama.cpp llama-server, Ollama /v1, vLLM, KoboldCpp,
   OpenRouter, OpenAI, Gemini's OpenAI-compatible endpoint, ...)
- "cli": any command-line model runner. The full prompt is piped to the
  command's stdin and stdout is taken as the answer
  (Claude Code: `claude -p --output-format text`, Gemini CLI: `gemini -p`, ...)
"""

import json
import os
import shlex
import subprocess
import urllib.request
import urllib.error


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
        raise LLMError(f"LLM server returned HTTP {e.code}: {body}") from e
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

#: Idle seconds per mode. There is no unload endpoint the four supported
#: backends share, but both servers that can unload at all take a time-to-live
#: on the request itself — LM Studio as `ttl`, Ollama as `keep_alive` — so the
#: policy rides along with the generation instead of needing a second call.
#: llama.cpp and vLLM hold one model for the life of the process; they ignore
#: both fields, which is the correct behaviour there rather than a failure.
_TTL_SECONDS = {"5m": 300, "now": 1}


def unload_payload(mode):
    """The keep-alive fields for `mode`, or {} when the model should stay put."""
    if mode not in _TTL_SECONDS:
        return {}
    seconds = _TTL_SECONDS[mode]
    return {"ttl": seconds, "keep_alive": 0 if mode == "now" else f"{seconds // 60}m"}


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


def call_openai_compatible(base_url, model, api_key, system_prompt, user_text,
                           images_base64=None, temperature=0.7, seed=-1, timeout=600,
                           max_tokens=DEFAULT_MAX_TOKENS, thinking="auto", unload_after="keep"):
    url = base_url.rstrip("/") + "/chat/completions"
    user_text = apply_thinking(user_text, thinking)

    if images_base64:
        content = [{"type": "text", "text": user_text}]
        for b64 in images_base64:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}})
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
    payload.update(unload_payload(unload_after))
    if thinking in ("off", "on"):
        # vLLM and recent LM Studio builds switch Qwen3's chat template with
        # this. It is the reliable half of the switch where it is supported;
        # the /no_think token already in user_text covers everywhere else.
        payload["chat_template_kwargs"] = {"enable_thinking": thinking == "on"}

    def post():
        return _post_json(url, payload, api_key, timeout)

    try:
        return _extract_text(post())
    except LLMError as first:
        msg = str(first).lower()
        # A server that does not know chat_template_kwargs may reject the whole
        # request. Drop it and let the text token carry the intent alone.
        # A server that does not know one of the optional fields may reject the
        # whole request. Shed them and retry on the essentials.
        optional = [k for k in ("chat_template_kwargs", "ttl", "keep_alive") if k in payload]
        if optional and ("400" in msg or "template" in msg or "extra" in msg
                         or "unrecognized" in msg or "unknown" in msg):
            for k in optional:
                payload.pop(k)
            return _extract_text(post())
        # Only a vision rejection is worth a second call. Retrying a dead server
        # or a 401 just doubles the wait (and, on a paid endpoint, the bill).
        if images_base64 and ("image" in msg or "vision" in msg or "content" in msg or "400" in msg):
            payload["messages"][1]["content"] = user_text
            return _extract_text(post())
        raise


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

# Backend presets — pick one and the standard address/command fills itself in.
# (Same approach as ComfyUI-LLM-Hub: aliases share the OpenAI-compatible client,
#  they only differ in the default port.)
PRESET_BASE_URLS = {
    "lmstudio": "http://127.0.0.1:1234/v1",
    "ollama": "http://127.0.0.1:11434/v1",
    "llamacpp": "http://127.0.0.1:8080/v1",
    "vllm": "http://127.0.0.1:8000/v1",
}
PRESET_CLI_COMMANDS = {
    "claude_cli": "claude -p --output-format text",
    "gemini_cli": "gemini -p",
    "codex_cli": "codex exec",
}
HTTP_BACKENDS = ["lmstudio", "ollama", "llamacpp", "vllm", "openai_compat"]
CLI_BACKEND_LIST = ["claude_cli", "gemini_cli", "codex_cli", "custom_cli"]
BACKEND_NAMES = HTTP_BACKENDS + CLI_BACKEND_LIST

# v1.0 saved workflows used these names
_BACKEND_ALIASES = {"openai_compatible": "openai_compat", "cli": "custom_cli"}


def normalize_backend(name):
    return _BACKEND_ALIASES.get(name, name)


_MODEL_CACHE = {"at": 0.0, "ids": []}
_MODEL_CACHE_TTL = 30.0


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
    for url in PRESET_BASE_URLS.values():
        try:
            req = urllib.request.Request(url.rstrip("/") + "/models")
            with urllib.request.urlopen(req, timeout=timeout) as res:
                data = json.loads(res.read().decode("utf-8", errors="replace"))
            for m in data.get("data", []) or []:
                mid = m.get("id") if isinstance(m, dict) else None
                if mid and mid not in ids:
                    ids.append(mid)
        except Exception:
            continue
    _MODEL_CACHE["at"] = now
    _MODEL_CACHE["ids"] = ids
    return list(ids)


def resolve_backend(backend, base_url, model, cli_command, server_model=AUTO_MODEL):
    """Turn the widget values into a concrete (kind, url, model, command)."""
    backend = normalize_backend(backend)
    if backend in CLI_BACKEND_LIST:
        cmd = (cli_command or "").strip() or PRESET_CLI_COMMANDS.get(backend, "")
        if not cmd:
            raise LLMError(f"Backend '{backend}' needs cli_command (e.g. 'claude -p --output-format text').")
        return ("cli", "", "", cmd)
    url = (base_url or "").strip() or PRESET_BASE_URLS.get(backend, "")
    if not url:
        raise LLMError(f"Backend '{backend}' needs base_url (an OpenAI-compatible /v1 address).")
    mdl = model
    if server_model and server_model != AUTO_MODEL:
        mdl = server_model
    return ("http", url, mdl, "")


def resolve_api_key(api_key):
    """Prefer the typed key, else the environment — so a shared workflow need not carry one."""
    if (api_key or "").strip():
        return api_key.strip()
    for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "H3_LLM_API_KEY"):
        v = os.environ.get(var, "").strip()
        if v:
            return v
    return ""


def call_llm(backend, base_url, model, api_key, cli_command, system_prompt,
             user_text, images_base64=None, temperature=0.7, seed=-1, timeout=600,
             server_model=AUTO_MODEL, max_tokens=DEFAULT_MAX_TOKENS, thinking="auto",
             unload_after="keep"):
    kind, url, mdl, cmd = resolve_backend(backend, base_url, model, cli_command, server_model)
    if kind == "cli":
        # A CLI runner has no template switch; the text token is all there is.
        return call_cli(cmd, system_prompt, apply_thinking(user_text, thinking), timeout=timeout)
    api_key = resolve_api_key(api_key)
    return call_openai_compatible(url, mdl, api_key, system_prompt, user_text,
                                  images_base64=images_base64, temperature=temperature,
                                  seed=seed, timeout=timeout, max_tokens=max_tokens,
                                  thinking=thinking, unload_after=unload_after)


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
    key = resolve_api_key(api_key)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            payload = json.loads(res.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        hint = " API 키를 확인하세요." if exc.code in (401, 403) else ""
        return {"ok": False, "kind": "http", "models": [], "target": url,
                "detail": f"HTTP {exc.code} {exc.reason}.{hint}"}
    except Exception as exc:
        return {"ok": False, "kind": "http", "models": [], "target": url,
                "detail": f"{url} 에 연결하지 못했습니다 ({type(exc).__name__}). "
                          f"서버가 켜져 있고 주소가 맞는지 확인하세요."}

    models = []
    for m in (payload.get("data") or []):
        mid = m.get("id") if isinstance(m, dict) else None
        if mid and mid not in models:
            models.append(mid)
    return {"ok": True, "kind": "http", "models": models, "target": url,
            "detail": f"모델 {len(models)}개 확인됨"}


def warm_up_model(backend, base_url, api_key, model, timeout=180.0):
    """Make the server load `model` into memory now, rather than mid-generation.

    LM Studio, Ollama and vLLM all load a model lazily on its first request, so
    the first real prompt otherwise pays a stall long enough to look like a
    hang. A one-token completion is the portable way to trigger that — there is
    no load endpoint the four backends share.

    Returns {ok, detail}. Never raises.
    """
    if not model:
        return {"ok": False, "detail": "로드할 모델을 먼저 선택하세요."}
    try:
        _kind, url, _m, _c = resolve_backend(backend, base_url, model, "")
    except LLMError as exc:
        return {"ok": False, "detail": str(exc)}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    try:
        _post_json(url.rstrip("/") + "/chat/completions", body, resolve_api_key(api_key), timeout)
    except Exception as exc:
        return {"ok": False, "detail": f"{model} 로드 실패: {exc}"}
    return {"ok": True, "detail": f"{model} 메모리에 로드됨"}
