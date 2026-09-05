#!/usr/bin/env python3
"""Run: python3 tests/test_gemini_backend.py

The gemini preset must reach Google's pinned host with Google's own key and
the right request shape — and never let the generic OpenAI key, the Qwen
/no_think token, or a VRAM warm-up call leak into a Google request.

(auto) must come from Google's live list, never from a name fixed in the
code: the hard-coded gemini-2.5-flash default turned into a 404 for every
user the day Google retired it (2026-06-17)."""
import importlib.util
import io
import json
import os
import pathlib
import sys
import time
import urllib.error

P = pathlib.Path(__file__).resolve().parent.parent
sp = importlib.util.spec_from_file_location(
    "h3g", P / "__init__.py", submodule_search_locations=[str(P)])
m = importlib.util.module_from_spec(sp)
sys.modules["h3g"] = m
sp.loader.exec_module(m)
import importlib
L = importlib.import_module("h3g.llm_backends")

passed, fails = 0, []


def ok(n, c, d=""):
    global passed
    if c:
        passed += 1
    else:
        fails.append(f"{n}{chr(10) + '      ' + str(d) if d else ''}")


def reset_default_cache():
    L._GEMINI_DEFAULT_CACHE.update({"at": 0.0, "key": "", "id": ""})


# --- wiring -----------------------------------------------------------------
ok("gemini is a selectable backend", "gemini" in L.BACKEND_NAMES)
ok("preset address is Google's OpenAI-compatible endpoint",
   L.PRESET_BASE_URLS["gemini"] == "https://generativelanguage.googleapis.com/v1beta/openai")
ok("local discovery never probes the cloud",
   L.GEMINI_BASE_URL not in L.LOCAL_PRESET_BASE_URLS.values())
ok("gemini gets no ttl/keep_alive dialect", L.unload_payload("now", "gemini") == {})

kind, url, mdl, _cmd = L.resolve_backend("gemini", "", "", "")
ok("empty model stays empty here — (auto) is resolved against the live list later",
   (kind, url, mdl) == ("http", L.GEMINI_BASE_URL, ""), (kind, url, mdl))
_k, _u, mdl, _c = L.resolve_backend("gemini", "", "models/gemini-3.6-flash", "")
ok("models/ prefix from Google's list is stripped", mdl == "gemini-3.6-flash", mdl)
_k, _u, mdl, _c = L.resolve_backend("gemini", "", "typed-model", "", server_model="gemini-3.8-flash")
ok("server_model dropdown wins over the typed field", mdl == "gemini-3.8-flash", mdl)

# --- model list filtering ----------------------------------------------------
listing = ["models/gemini-3.6-flash", "models/gemini-3.1-pro",
           "models/gemini-embedding-001", "models/text-embedding-004",
           "models/imagen-3.0-generate-002", "models/gemini-3.6-flash-image",
           "models/veo-2.0-generate-001", "models/gemini-3.6-flash-preview-tts",
           "models/gemini-3.6-flash-live-001", "models/aqa",
           "models/gemma-3-27b-it", "models/gemini-3.6-flash"]
ok("only chat-capable ids survive, bare and deduped",
   L.filter_gemini_chat_models(listing) ==
   ["gemini-3.6-flash", "gemini-3.1-pro", "gemma-3-27b-it"],
   L.filter_gemini_chat_models(listing))

# --- (auto) picks the newest stable Flash from whatever Google lists --------
ok("newest stable flash wins over older stable, previews, lite and pro",
   L.pick_default_gemini_model([
       "gemini-2.5-flash", "gemini-3.6-flash", "gemini-3.8-flash-preview",
       "gemini-3.6-flash-lite", "gemini-3.1-pro", "gemini-3-flash-preview"]) == "gemini-3.6-flash")
ok("version compare is numeric, not lexical (3.10 > 3.6)",
   L.pick_default_gemini_model(["gemini-3.6-flash", "gemini-3.10-flash"]) == "gemini-3.10-flash")
ok("a preview is chosen only when no stable flash exists",
   L.pick_default_gemini_model(["gemini-3-flash-preview", "gemini-3.8-flash-preview-05-20",
                                "gemini-3.1-pro"]) == "gemini-3.8-flash-preview-05-20")
ok("no flash at all -> the fallback constant",
   L.pick_default_gemini_model(["gemini-3.1-pro", "gemma-3-27b-it"]) == L.DEFAULT_GEMINI_MODEL)
ok("a preview with a product suffix is a different product, never (auto)",
   L.pick_default_gemini_model(["gemini-2.0-flash-preview-image-generation"]) == L.DEFAULT_GEMINI_MODEL)
ok("empty list -> the fallback constant",
   L.pick_default_gemini_model([]) == L.DEFAULT_GEMINI_MODEL)
ok("the roomy-free-tier 3.5 flash is preferred over newer flashes (20 RPD free)",
   L.pick_default_gemini_model(["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.6-flash"]) == "gemini-3.5-flash")
ok("a 3.5 preview still beats a newer stable when no 3.5 stable is listed",
   L.pick_default_gemini_model(["gemini-3.6-flash", "gemini-3.5-flash-preview-05-20"])
   == "gemini-3.5-flash-preview-05-20")
os.environ[L.GEMINI_DEFAULT_MODEL_ENV] = "models/gemini-3.7-flash"
try:
    ok("H3_GEMINI_DEFAULT_MODEL pins (auto), models/ prefix tolerated",
       L.pick_default_gemini_model(["gemini-3.5-flash"]) == "gemini-3.7-flash")
finally:
    os.environ.pop(L.GEMINI_DEFAULT_MODEL_ENV, None)

# --- native list: capability flags + paging, with the OpenAI list as fallback
native_pages = {
    "": {"models": [
        {"name": "models/gemini-3.6-flash", "supportedGenerationMethods": ["generateContent", "countTokens"]},
        {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
        {"name": "models/gemini-3.6-flash-image", "supportedGenerationMethods": ["generateContent"]},
    ], "nextPageToken": "p2"},
    "p2": {"models": [
        {"name": "models/gemini-3.8-flash-preview", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/veo-3.0", "supportedGenerationMethods": ["predictLongRunning"]},
    ]},
}
seen_urls = []


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_urlopen(req, timeout=None):
    url = req.full_url if hasattr(req, "full_url") else str(req)
    seen_urls.append(url)
    if url.startswith(L.GEMINI_NATIVE_MODELS_URL):
        token = ""
        if "pageToken=" in url:
            token = url.split("pageToken=", 1)[1]
        return _Resp(json.dumps(native_pages[token]).encode())
    raise AssertionError(f"unexpected fetch: {url}")


orig_urlopen = urllib.request.urlopen
L.urllib.request.urlopen = fake_urlopen
try:
    seen_urls.clear()
    got = L.gemini_chat_models("k")
    ok("native list is read by capability, across pages, and name-filtered",
       got == ["gemini-3.6-flash", "gemini-3.8-flash-preview"], got)
    ok("native list sends the key as x-goog-api-key and pages with pageSize=1000",
       len(seen_urls) == 2 and "pageSize=1000" in seen_urls[0] and "pageToken=p2" in seen_urls[1],
       seen_urls)

    reset_default_cache()
    seen_urls.clear()
    ok("(auto) resolves to the newest stable flash on the live list",
       L.default_gemini_model("k") == "gemini-3.6-flash")
    first_calls = len(seen_urls)
    L.default_gemini_model("k")
    ok("(auto) is cached per key, not fetched on every generation",
       len(seen_urls) == first_calls, (first_calls, len(seen_urls)))
finally:
    L.urllib.request.urlopen = orig_urlopen
    reset_default_cache()


def failing_urlopen(req, timeout=None):
    raise urllib.error.URLError("dns down")


L.urllib.request.urlopen = failing_urlopen
try:
    got = L.gemini_chat_models("k", fallback_ids=["models/gemini-3.6-flash", "models/aqa"])
    ok("native list unreachable -> the OpenAI-shaped list is filtered instead",
       got == ["gemini-3.6-flash"], got)
    reset_default_cache()
    ok("(auto) without any list falls back to the constant, uncached",
       L.default_gemini_model("k") == L.DEFAULT_GEMINI_MODEL
       and L._GEMINI_DEFAULT_CACHE["id"] == "")
finally:
    L.urllib.request.urlopen = orig_urlopen
    reset_default_cache()

# --- Google's error body reaches the overlay; other remote bodies do not -----
def http_error_urlopen(code, body):
    def _open(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, code, "Err", {}, io.BytesIO(body.encode()))
    return _open


google_404 = json.dumps({"error": {"code": 404, "status": "NOT_FOUND",
                                   "message": "models/gemini-2.5-flash is not found"}})
L.urllib.request.urlopen = http_error_urlopen(404, google_404)
try:
    try:
        L._post_json(L.GEMINI_BASE_URL + "/chat/completions", {}, "k", 5)
        ok("google 404 raises", False)
    except L.LLMError as exc:
        msg = str(exc)
        ok("google error body is shown", "gemini-2.5-flash is not found" in msg, msg)
        ok("google 404 carries the retired-model hint", "연결 확인" in msg and "(auto)" in msg, msg)
    try:
        L._post_json("https://example.com/v1/chat/completions", {}, "k", 5)
        ok("other remote 404 raises", False)
    except L.LLMError as exc:
        ok("other remote hosts still get only the status line",
           "not found" not in str(exc).lower() and "404" in str(exc), str(exc))
finally:
    L.urllib.request.urlopen = orig_urlopen

quota0 = "Quota exceeded for metric: generate_content_free_tier_requests, limit: 0, model: gemini-3.1-pro"
ok("429 with limit 0 -> no free tier hint", "무료 등급이 없습니다" in L.gemini_error_hint(429, quota0))
ok("429 per-day quota -> daily hint",
   "일일" in L.gemini_error_hint(429, '{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}'))
ok("429 otherwise -> per-minute hint", "분당" in L.gemini_error_hint(429, "rate"))
ok("400 bad key -> key hint", "aistudio.google.com/apikey" in L.gemini_error_hint(400, "API key not valid"))
ok("403 -> permission hint", "권한" in L.gemini_error_hint(403, "PERMISSION_DENIED"))
ok("unknown -> no hint", L.gemini_error_hint(500, "boom") == "")

# --- env key routing ---------------------------------------------------------
SAVED = {k: os.environ.pop(k, None) for k in
         ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY",
          "GOOGLE_API_KEY", "H3_LLM_API_KEY", "H3_LLM_ALLOWED_HOSTS")}
try:
    os.environ["OPENAI_API_KEY"] = "sk-openai"
    os.environ["GEMINI_API_KEY"] = "goog-key"
    ok("google host gets the Gemini key, not OPENAI_API_KEY",
       L.resolve_api_key("", L.GEMINI_BASE_URL) == "goog-key")
    ok("loopback still prefers OPENAI_API_KEY",
       L.resolve_api_key("", "http://127.0.0.1:1234/v1") == "sk-openai")
    ok("typed key always wins",
       L.resolve_api_key(" typed ", L.GEMINI_BASE_URL) == "typed")
    del os.environ["GEMINI_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = "goog2"
    ok("GOOGLE_API_KEY is the fallback spelling",
       L.resolve_api_key("", L.GEMINI_BASE_URL) == "goog2")
    del os.environ["GOOGLE_API_KEY"]
    ok("no Google key -> nothing travels (the OPENAI key stays home)",
       L.resolve_api_key("", L.GEMINI_BASE_URL) == "")
    ok("other remote hosts still need the allowlist",
       L.resolve_api_key("", "https://example.com/v1") == "")
finally:
    for k, v in SAVED.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v

# --- wire shape, via a captured _post_json -----------------------------------
sent, script = [], []


def fake_post(url, payload, api_key, timeout):
    sent.append({"url": url, "payload": json.loads(json.dumps(payload)), "key": api_key})
    step = script.pop(0) if script else None
    if step:
        raise L.LLMError(step)
    return {"choices": [{"message": {"content": "final prompt body"}}]}


orig_post = L._post_json
orig_native = L._gemini_native_models
L._post_json = fake_post
L._gemini_native_models = lambda api_key, timeout: [
    "models/gemini-3.6-flash", "models/gemini-3.8-flash-preview", "models/gemini-3.1-pro"]
reset_default_cache()
try:
    def call(thinking="auto", steps=()):
        sent.clear()
        script.clear()
        script.extend(steps)
        return L.call_llm("gemini", "", "", "", "", "sys", "장면 요청",
                          thinking=thinking, max_tokens=60000, unload_after="now")

    call("auto")
    b = sent[0]["payload"]
    ok("request goes to Google's chat endpoint",
       sent[0]["url"] == L.GEMINI_BASE_URL + "/chat/completions", sent[0]["url"])
    ok("(auto) sends the live newest stable flash, not a name from the code",
       b["model"] == "gemini-3.6-flash", b["model"])
    ok("auto sends no thinking fields",
       "reasoning_effort" not in b and "chat_template_kwargs" not in b)
    ok("no ttl/keep_alive to Google", "ttl" not in b and "keep_alive" not in b)

    call("off")
    b = sent[0]["payload"]
    ok("off maps to reasoning_effort low (3.x models reject none)",
       b.get("reasoning_effort") == "low", b.get("reasoning_effort"))
    ok("off never appends /no_think for Gemini",
       not b["messages"][1]["content"].endswith("/no_think"), b["messages"][1]["content"])
    ok("off sends no Qwen template switch", "chat_template_kwargs" not in b)

    call("on")
    ok("on maps to reasoning_effort high",
       sent[0]["payload"].get("reasoning_effort") == "high")

    # A model that does not take the field at all rejects it with a 400.
    call("off", steps=("Gemini API returned HTTP 400: reasoning_effort is not supported",))
    ok("rejected reasoning_effort is shed and retried",
       len(sent) == 2 and "reasoning_effort" not in sent[1]["payload"], len(sent))

    # A model whose output cap sits under the 60000 default — Google's wording.
    call("auto", steps=('Gemini API returned HTTP 400: {"error": {"message": '
                        '"max_output_tokens must be less than or equal to 8192"}}',))
    ok("rejected max_tokens is dropped so the model's own cap applies",
       len(sent) == 2 and "max_tokens" not in sent[1]["payload"]
       and sent[1]["payload"]["messages"] == sent[0]["payload"]["messages"], len(sent))

    # Local servers keep the old contract untouched.
    sent.clear()
    script.clear()
    L.call_llm("openai_compat", "http://127.0.0.1:9999/v1", "m", "", "", "sys", "장면 요청",
               thinking="off", max_tokens=60000, unload_after="keep")
    b = sent[0]["payload"]
    ok("local off still uses /no_think + template switch",
       b["messages"][1]["content"].endswith("/no_think")
       and b.get("chat_template_kwargs") == {"enable_thinking": False})
finally:
    L._post_json = orig_post
    L._gemini_native_models = orig_native
    reset_default_cache()


# --- warm-up must not spend a real (billable) request ------------------------
def boom(*a, **k):
    raise AssertionError("warm-up must not call the API for gemini")


orig_post = L._post_json
L._post_json = boom
try:
    r = L.warm_up_model("gemini", "", "", "gemini-3.6-flash")
finally:
    L._post_json = orig_post
ok("warm-up is a no-op for the cloud API", r["ok"] is True, r)

# --- safety settings + empty-answer diagnosis --------------------------------
payload, _text, is_g = L._chat_payload(L.GEMINI_BASE_URL, "gemini-3.6-flash", "sys", "장면",
                                       None, None, 0.7, -1, 60000, "off", "keep", "gemini")
ok("gemini requests carry AI Studio's safety-off settings",
   is_g and payload.get("extra_body") == {"google": {"safety_settings": [
       {"category": c, "threshold": "BLOCK_NONE"} for c in L.GEMINI_SAFETY_CATEGORIES]}},
   payload.get("extra_body"))
payload, _text, is_g = L._chat_payload("http://127.0.0.1:1234/v1", "m", "sys", "장면",
                                       None, None, 0.7, -1, 60000, "off", "keep", "lmstudio")
ok("local requests carry no google extension", not is_g and "extra_body" not in payload)

payload = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
           "extra_body": {"google": {}}, "max_tokens": 60000}
ok("a server that rejects extra_body is retried without it",
   L._retry_chat_payload(payload, 'Gemini API returned HTTP 400: Unknown name "extra_body"', True, "u")
   and "extra_body" not in payload, payload)

try:
    L._extract_text({"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]})
    ok("empty non-streamed answer raises", False)
except L.LLMError as exc:
    ok("empty answer names the finish reason and the safety filter",
       "finish_reason=content_filter" in str(exc) and "안전 필터" in str(exc), str(exc))

sse = [
    b'data: {"choices":[{"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n',
    b'data: {"choices":[{"delta":{},"finish_reason":"content_filter"}]}\n',
    b'data: [DONE]\n',
]
try:
    L._read_openai_stream(iter(sse), None, None, time.perf_counter())
    ok("empty stream raises", False)
except L.LLMError as exc:
    ok("empty stream names the finish reason and the safety filter",
       "finish_reason=content_filter" in str(exc) and "안전 필터" in str(exc), str(exc))

sse_len = [b'data: {"choices":[{"delta":{"content":""},"finish_reason":"length"}]}\n',
           b'data: [DONE]\n']
try:
    L._read_openai_stream(iter(sse_len), None, None, time.perf_counter())
    ok("length-cut empty stream raises", False)
except L.LLMError as exc:
    ok("a token-budget cut is reported as such",
       "finish_reason=length" in str(exc) and "토큰" in str(exc), str(exc))

sse_ok = [b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n',
          b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
          b'data: [DONE]\n']
answer, metrics = L._read_openai_stream(iter(sse_ok), None, None, time.perf_counter())
ok("a normal stream still returns its text", answer == "hello" and metrics["streamed"] is True)

if fails:
    print("GEMINI BACKEND TESTS FAILED")
    for f in fails:
        print(" -", f)
    raise SystemExit(1)
print(f"✓ {passed} passed, 0 failed")
