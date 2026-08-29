#!/usr/bin/env python3
"""Run: python3 tests/test_gemini_backend.py

The gemini preset must reach Google's pinned host with Google's own key and
the right request shape — and never let the generic OpenAI key, the Qwen
/no_think token, or a VRAM warm-up call leak into a Google request."""
import importlib.util
import json
import os
import pathlib
import sys

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


# --- wiring -----------------------------------------------------------------
ok("gemini is a selectable backend", "gemini" in L.BACKEND_NAMES)
ok("preset address is Google's OpenAI-compatible endpoint",
   L.PRESET_BASE_URLS["gemini"] == "https://generativelanguage.googleapis.com/v1beta/openai")
ok("local discovery never probes the cloud",
   L.GEMINI_BASE_URL not in L.LOCAL_PRESET_BASE_URLS.values())
ok("gemini gets no ttl/keep_alive dialect", L.unload_payload("now", "gemini") == {})

kind, url, mdl, _cmd = L.resolve_backend("gemini", "", "", "")
ok("empty model falls back to the default",
   (kind, url, mdl) == ("http", L.GEMINI_BASE_URL, L.DEFAULT_GEMINI_MODEL), (kind, url, mdl))
_k, _u, mdl, _c = L.resolve_backend("gemini", "", "models/gemini-2.5-pro", "")
ok("models/ prefix from Google's list is stripped", mdl == "gemini-2.5-pro", mdl)
_k, _u, mdl, _c = L.resolve_backend("gemini", "", "typed-model", "", server_model="gemini-3-flash")
ok("server_model dropdown wins over the typed field", mdl == "gemini-3-flash", mdl)

# --- model list filtering ----------------------------------------------------
listing = ["models/gemini-2.5-flash", "models/gemini-2.5-pro",
           "models/gemini-embedding-001", "models/text-embedding-004",
           "models/imagen-3.0-generate-002", "models/gemini-2.5-flash-image",
           "models/veo-2.0-generate-001", "models/gemini-2.5-flash-preview-tts",
           "models/gemini-2.0-flash-live-001", "models/aqa",
           "models/gemma-3-27b-it", "models/gemini-2.5-flash"]
ok("only chat-capable ids survive, bare and deduped",
   L.filter_gemini_chat_models(listing) ==
   ["gemini-2.5-flash", "gemini-2.5-pro", "gemma-3-27b-it"],
   L.filter_gemini_chat_models(listing))

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
L._post_json = fake_post
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
    ok("default model fills in", b["model"] == L.DEFAULT_GEMINI_MODEL, b["model"])
    ok("auto sends no thinking fields",
       "reasoning_effort" not in b and "chat_template_kwargs" not in b)
    ok("no ttl/keep_alive to Google", "ttl" not in b and "keep_alive" not in b)

    call("off")
    b = sent[0]["payload"]
    ok("off maps to reasoning_effort none",
       b.get("reasoning_effort") == "none", b.get("reasoning_effort"))
    ok("off never appends /no_think for Gemini",
       not b["messages"][1]["content"].endswith("/no_think"), b["messages"][1]["content"])
    ok("off sends no Qwen template switch", "chat_template_kwargs" not in b)

    call("on")
    ok("on maps to reasoning_effort high",
       sent[0]["payload"].get("reasoning_effort") == "high")

    # A model that cannot switch thinking off (2.5-pro) rejects the field by name.
    call("off", steps=("LLM server returned HTTP 400: reasoning_effort is not supported",))
    ok("rejected reasoning_effort is shed and retried",
       len(sent) == 2 and "reasoning_effort" not in sent[1]["payload"], len(sent))

    # A model whose output cap sits under the 60000 default.
    call("auto", steps=("LLM server returned HTTP 400: max_tokens must be at most 8192",))
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


# --- warm-up must not spend a real (billable) request ------------------------
def boom(*a, **k):
    raise AssertionError("warm-up must not call the API for gemini")


orig_post = L._post_json
L._post_json = boom
try:
    r = L.warm_up_model("gemini", "", "", "gemini-2.5-flash")
finally:
    L._post_json = orig_post
ok("warm-up is a no-op for the cloud API", r["ok"] is True, r)

if fails:
    print("GEMINI BACKEND TESTS FAILED")
    for f in fails:
        print(" -", f)
    raise SystemExit(1)
print(f"✓ {passed} passed, 0 failed")
