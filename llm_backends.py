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


def call_openai_compatible(base_url, model, api_key, system_prompt, user_text,
                           images_base64=None, temperature=0.7, seed=-1, timeout=600,
                           max_tokens=8192):
    url = base_url.rstrip("/") + "/chat/completions"

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

    try:
        data = _post_json(url, payload, api_key, timeout)
    except LLMError as first:
        # Only a vision rejection is worth a second call. Retrying a dead server or a
        # 401 just doubles the wait (and, on a paid endpoint, the bill).
        msg = str(first).lower()
        retryable = images_base64 and ("image" in msg or "vision" in msg or "content" in msg or "400" in msg)
        if not retryable:
            raise
        payload["messages"][1]["content"] = user_text
        data = _post_json(url, payload, api_key, timeout)

    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise LLMError(f"Unexpected LLM response shape: {str(data)[:500]}")
    if not text or not text.strip():
        raise LLMError("LLM returned an empty response.")
    return text


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
             server_model=AUTO_MODEL, max_tokens=8192):
    kind, url, mdl, cmd = resolve_backend(backend, base_url, model, cli_command, server_model)
    if kind == "cli":
        return call_cli(cmd, system_prompt, user_text, timeout=timeout)
    api_key = resolve_api_key(api_key)
    return call_openai_compatible(url, mdl, api_key, system_prompt, user_text,
                                  images_base64=images_base64, temperature=temperature,
                                  seed=seed, timeout=timeout, max_tokens=max_tokens)
