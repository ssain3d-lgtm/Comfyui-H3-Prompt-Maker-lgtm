#!/usr/bin/env python3
"""Run: python3 tests/test_ninfer.py

NInfer (`ninfer-serve`) as a preset backend.

Unlike the other presets this one was checked against a real server
(Qwen3.8-27B NVFP4 on an RTX 5090, 2026-09). What that measurement changed:

  /no_think does NOT switch thinking off on NInfer. The model reads it as part
  of the prompt -- reasoning got *longer* (51 -> 144 characters on "What is
  2+2?") and the answer got worse. chat_template_kwargs alone does switch it,
  both ways, so for NInfer the text token is pure prompt pollution.

  reasoning_effort and chat_template_kwargs only collide when they disagree
  ("none" + enable_thinking true -> HTTP 400 conflicting_template_option), so
  the existing single mechanism is safe to keep.

  NInfer accepts any max_tokens, 1,000,000 included, and clamps internally.
  There is nothing to guard against, so this backend gets no special cap.
"""
import importlib.util
import io
import json
import pathlib
import sys
import threading
from contextlib import redirect_stderr
from http.server import BaseHTTPRequestHandler, HTTPServer

P = pathlib.Path(__file__).resolve().parent.parent
sp = importlib.util.spec_from_file_location(
    "h3ninfer", P / "__init__.py", submodule_search_locations=[str(P)])
m = importlib.util.module_from_spec(sp)
sys.modules["h3ninfer"] = m
sp.loader.exec_module(m)
L = importlib.import_module("h3ninfer.llm_backends")

passed, failures = 0, []


def ok(name, cond, detail=""):
    global passed
    if cond:
        passed += 1
    else:
        failures.append(f"{name}{chr(10) + '      ' + str(detail) if detail else ''}")


# A server that records what it was asked, so "what goes over the wire" is the
# assertion rather than "what the code looks like".
seen, hits = [], []


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, payload):
        b = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        hits.append(self.path)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        seen.append(body)
        self._json(200, {"choices": [{"message": {"content":
                   "subject_definitions: x\nsummary: y\ndetailed_description: z"}}]})


srv = HTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{srv.server_port}/v1"


def call(mode, unload="keep", backend="ninfer"):
    seen.clear()
    hits.clear()
    L.call_llm(backend, URL, "qwen3.8-27b", "", "", "sys", "장면 요청",
               thinking=mode, max_tokens=60000, unload_after=unload)
    return seen[0]


# -- the preset itself ------------------------------------------------------

ok("ninfer is a selectable backend", "ninfer" in L.BACKEND_NAMES)
ok("its preset address is NInfer's standard port",
   L.LOCAL_PRESET_BASE_URLS.get("ninfer") == "http://127.0.0.1:8081/v1",
   L.LOCAL_PRESET_BASE_URLS.get("ninfer"))
ok("choosing it needs no typed address",
   L.resolve_backend("ninfer", "", "qwen3.8-27b", "") == ("http", "http://127.0.0.1:8081/v1", "qwen3.8-27b", ""))
ok("the existing backends keep their names",
   L.BACKEND_NAMES[:6] == ["lmstudio", "ollama", "llamacpp", "vllm", "gemini", "openai_compat"],
   L.BACKEND_NAMES)
ok("model discovery looks at 8081 too",
   "http://127.0.0.1:8081/v1" in L.LOCAL_PRESET_BASE_URLS.values())

# -- thinking: the template switch, never the text token --------------------

for mode in ("auto", "off", "on"):
    body = call(mode)
    text = body["messages"][1]["content"]
    ok(f"{mode}: no /no_think or /think lands in the prompt",
       "/no_think" not in text and "/think" not in text, repr(text))

ok("auto: nothing is sent", "chat_template_kwargs" not in call("auto"))
ok("off: the template switch says enable_thinking false",
   call("off").get("chat_template_kwargs") == {"enable_thinking": False})
ok("on: the template switch says enable_thinking true",
   call("on").get("chat_template_kwargs") == {"enable_thinking": True})
ok("one mechanism only -- reasoning_effort would collide with it",
   "reasoning_effort" not in call("off"))

# The text token is the portable half for servers that ignore what they do not
# know. NInfer is the exception, so the other backends must keep it.
other = call("off", backend="openai_compat")
ok("openai_compat still gets the text token",
   "/no_think" in other["messages"][1]["content"])
ok("llamacpp still gets the text token",
   "/no_think" in call("off", backend="llamacpp")["messages"][1]["content"])

# -- unload: say what is true, do not pretend -------------------------------

ok("no keep-alive dialect is invented for NInfer",
   L.unload_payload("now", "ninfer") == {} and L.unload_payload("5m", "ninfer") == {})

for mode in ("now", "5m"):
    err = io.StringIO()
    with redirect_stderr(err):
        call("off", unload=mode)
    note = err.getvalue()
    ok(f"unload_after={mode}: no unload request is fired at NInfer",
       hits == ["/v1/chat/completions"], hits)
    ok(f"unload_after={mode}: the reason is reported, not swallowed",
       "NInfer" in note and "stop" in note.lower(), repr(note))
    ok(f"unload_after={mode}: it does not claim the model was unloaded",
       "언로드됨" not in note, repr(note))

# "keep" promises nothing, and "close" promises an unload when the overlay closes,
# not after this generation. Only the two post-generation modes have something to
# explain -- warning on every generation for the others is just noise.
for mode in ("keep", "close"):
    err = io.StringIO()
    with redirect_stderr(err):
        call("off", unload=mode)
    ok(f"unload_after={mode}: stays quiet", err.getvalue() == "", repr(err.getvalue()))

result = L.unload_model("ninfer", URL, "", "qwen3.8-27b")
ok("the manual unload button reports NInfer as unsupported",
   result["ok"] is False and result["supported"] is False, result)
ok("and it names NInfer rather than a generic server",
   "NInfer" in result["detail"], result["detail"])

srv.shutdown()

for f in failures:
    print("  ✗ " + f)
print(f"{'✓' if not failures else '✗'} {passed} passed, {len(failures)} failed")
sys.exit(1 if failures else 0)
