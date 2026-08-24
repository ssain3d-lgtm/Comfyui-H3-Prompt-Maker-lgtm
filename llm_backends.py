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
                           images_base64=None, temperature=0.7, seed=-1, timeout=600):
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
    }
    if seed is not None and seed >= 0:
        payload["seed"] = int(seed)

    try:
        data = _post_json(url, payload, api_key, timeout)
    except LLMError:
        if not images_base64:
            raise
        # Vision payload rejected (text-only model loaded) -> retry text-only
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
    try:
        proc = subprocess.run(
            cli_command,
            shell=True,
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


def call_llm(backend, base_url, model, api_key, cli_command, system_prompt,
             user_text, images_base64=None, temperature=0.7, seed=-1, timeout=600):
    if backend == "cli":
        return call_cli(cli_command, system_prompt, user_text, timeout=timeout)
    return call_openai_compatible(base_url, model, api_key, system_prompt, user_text,
                                  images_base64=images_base64, temperature=temperature,
                                  seed=seed, timeout=timeout)
