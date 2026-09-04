"""Run: python3 tests/test_unload_reload.py

The unload/reload round trip, against a server that behaves like a real one:
it holds exactly one model in memory, drops it when told to, and answers 404
for a model it does not currently hold.

Both halves fail invisibly if they are wrong. An unload that never reaches the
server just leaves VRAM occupied, which nobody notices until the render OOMs.
A reload that never happens turns the next generate into a 404 that reads like
a broken address.
"""
import importlib.util
import json
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

P = pathlib.Path(__file__).resolve().parent.parent
sp = importlib.util.spec_from_file_location("h3u", P / "__init__.py", submodule_search_locations=[str(P)])
m = importlib.util.module_from_spec(sp); sys.modules["h3u"] = m; sp.loader.exec_module(m)
L = importlib.import_module("h3u.llm_backends")

MODEL = "qwen3.8-27b-uncensored"
state = {"loaded": None, "jit": True}
log = []


class Server(BaseHTTPRequestHandler):
    """One model in memory at a time, ttl/keep_alive honoured, JIT switchable."""

    def log_message(self, *a): pass

    def do_GET(self):
        if self.path.endswith("/api/v1/models"):
            loaded = ([{"id": MODEL, "config": {"context_length": 65536}}]
                      if state["loaded"] == MODEL else [])
            return self._send(200, {"models": [{
                "type": "llm", "key": MODEL, "selected_variant": MODEL,
                "variants": [MODEL], "architecture": "qwen3", "loaded_instances": loaded,
            }]})
        if self.path.endswith("/v1/models"):
            return self._send(200, {"data": [{"id": MODEL}]})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if self.path.endswith("/api/v1/models/load"):
            want = body.get("model")
            log.append({"kind": "load", "model": want})
            state["loaded"] = want
            return self._send(200, {"type": "llm", "instance_id": want, "status": "loaded"})
        if self.path.endswith("/api/v1/models/unload"):
            instance = body.get("instance_id")
            log.append({"kind": "unload", "model": instance})
            if state["loaded"] == instance:
                state["loaded"] = None
            return self._send(200, {"instance_id": instance})

        want = body.get("model")
        is_ping = body.get("max_tokens") == 1
        log.append({"kind": "ping" if is_ping else "gen", "model": want, "ping": is_ping, "ttl": body.get("ttl"),
                    "keep_alive": body.get("keep_alive"), "loaded_before": state["loaded"]})

        if state["loaded"] != want:
            if not state["jit"]:
                return self._send(404, {"error": {"message": f"Model '{want}' not found"}})
            state["loaded"] = want                       # JIT: load on demand

        # A manually loaded LM Studio model ignores a per-request ttl. This is
        # the real bug the native unload endpoint fixes. Ollama's keep_alive=0
        # remains an immediate instruction on the inference request itself.
        keep = body.get("keep_alive")
        drop = keep == 0
        out = {"choices": [{"message": {"content":
            "subject_definitions: a\nsummary: b\ndetailed_description: c"}}]}
        self._send(200, out)
        if drop:
            state["loaded"] = None

    def _send(self, code, payload):
        b = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


srv = HTTPServer(("127.0.0.1", 3404), Server)
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = "http://127.0.0.1:3404/v1"

passed, fails = 0, []


def ok(n, c, d=""):
    global passed
    if c:
        passed += 1
    else:
        fails.append(f"{n}{chr(10) + '      ' + str(d) if d else ''}")


def eq(n, a, b):
    ok(n, a == b, f"expected {b!r}, got {a!r}")


def generate(unload, backend="lmstudio"):
    log.clear()
    return L.call_llm(backend, URL, MODEL, "", "", "sys", "장면", max_tokens=60000,
                      unload_after=unload)


# --- the unload half --------------------------------------------------------
state["loaded"], state["jit"] = MODEL, True
generate("keep")
eq("keep: no ttl is sent", log[0]["ttl"], None)
eq("keep: no keep_alive is sent", log[0]["keep_alive"], None)
eq("keep: the model stays in memory", state["loaded"], MODEL)

state["loaded"] = MODEL
generate("5m")
eq("5m: ttl is 300 seconds", log[0]["ttl"], 300)
eq("5m: LM Studio is not sent Ollama's keep_alive", log[0]["keep_alive"], None)
eq("5m: the model is still there right after the answer", state["loaded"], MODEL)

state["loaded"] = MODEL
generate("now")
eq("now: ttl is the shortest the servers accept", log[0]["ttl"], 1)
eq("now: LM Studio is not sent Ollama's keep_alive", log[0]["keep_alive"], None)
eq("now: native unload follows generation", [e["kind"] for e in log], ["gen", "unload"])
eq("now: the model is actually gone afterwards", state["loaded"], None)

state["loaded"] = MODEL
generate("now", "ollama")
eq("ollama: no LM Studio ttl is sent", log[0]["ttl"], None)
eq("ollama: keep_alive=0 unloads immediately", log[0]["keep_alive"], 0)
eq("ollama: the model is gone afterwards", state["loaded"], None)

# --- the reload half --------------------------------------------------------
# This is the state the previous case leaves behind: nothing in memory.
eq("reload: memory really is empty going in", state["loaded"], None)
state["jit"] = True
out = generate("now")
ok("reload: a generate on an empty server still answers", "subject_definitions" in out, out[:60])
eq("reload: and it is the saved model that gets requested", log[0]["model"], MODEL)

# A server without JIT loading is the case that used to fail: nothing this code
# does would load the model, and the generate came back as a bare 404.
state["loaded"], state["jit"] = None, False
raised = None
try:
    generate("now")
except Exception as exc:  # noqa: BLE001
    raised = exc
ok("reload: a server that will not load on demand fails loudly", raised is not None)
ok("reload: and it is the model name that is reported, not a bare 404",
   raised is not None and MODEL in str(raised), str(raised)[:120])

# Warming first is what makes the reload this code's doing rather than a side
# effect of JIT. LM Studio v0.4 has a real load endpoint, so this succeeds even
# with JIT disabled.
state["loaded"], state["jit"] = None, False
warm = L.warm_up_model("lmstudio", URL, "", MODEL)
eq("warm: reports success", warm["ok"], True)
eq("warm: the model is in memory before any real call", state["loaded"], MODEL)
eq("warm: native load is used instead of a one-token inference", log[-1]["kind"], "load")

# --- through the route, which is what actually runs ------------------------
# The library doing the right thing is not the same as the route calling it.
# This drives generate_prompt itself and counts what the server was asked.
import asyncio

R = importlib.import_module("h3u.server_routes")


class FakeReq:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    @property
    def content(self):
        chunks = [self._b]

        class It:
            def iter_chunked(self_inner, _n):
                class A:
                    def __aiter__(self_a): self_a.i = 0; return self_a
                    async def __anext__(self_a):
                        if self_a.i >= len(chunks): raise StopAsyncIteration
                        self_a.i += 1
                        return chunks[self_a.i - 1]
                return A()
        return It()


def via_route(unload):
    log.clear()
    body = {"promptText": "골목", "minimaxStyle": "ref2va", "duration": 10,
            "llm": {"backend": "lmstudio", "base_url": URL, "model": MODEL,
                    "api_key": "", "cli_command": "", "temperature": 0.7,
                    "max_tokens": 60000, "thinking": "auto", "unload_after": unload}}
    routes = []

    class Table:
        def get(self, path):
            def deco(fn): routes.append((path, fn)); return fn
            return deco
        def post(self, path):
            def deco(fn): routes.append((path, fn)); return fn
            return deco

    R.register(Table())
    handler = next(fn for path, fn in routes if path.endswith("/generate-prompt"))
    # Keep this test dependency-free like the rest of the pack: server_routes
    # imports aiohttp only when constructing the response, so replace that tiny
    # boundary rather than requiring ComfyUI's environment in CI.
    original_json = R._json
    class Resp:
        def __init__(self, payload, status):
            self.text = json.dumps(payload)
            self.status = status
    R._json = lambda payload, status=200: Resp(payload, status)
    try:
        resp = asyncio.run(handler(FakeReq(body)))
    finally:
        R._json = original_json
    return json.loads(resp.text), [e["kind"] for e in log]


state["loaded"], state["jit"] = MODEL, True
out, seq = via_route("keep")
eq("route: keep makes exactly one call — no wasted warm-up", seq, ["gen"])
ok("route: keep still answers", "subject_definitions" in out.get("result", ""))

state["loaded"], state["jit"] = None, True
out, seq = via_route("now")
eq("route: an unload setting loads, generates, then really unloads", seq, ["load", "gen", "unload"])
ok("route: and the generate then succeeds", "subject_definitions" in out.get("result", ""))
eq("route: the model is unloaded again afterwards", state["loaded"], None)

out, seq = via_route("5m")
eq("route: 5m warms up too — the model may have aged out", seq, ["load", "gen"])
eq("route: and 5m leaves it loaded", state["loaded"], MODEL)

state["loaded"], state["jit"] = None, True
out, seq = via_route("close")
eq("route: close mode loads once and keeps the model for retries", seq, ["load", "gen"])
eq("route: close mode remains resident before the UI closes", state["loaded"], MODEL)
unloaded = L.unload_model("lmstudio", URL, "", MODEL)
eq("close: explicit unload reports success", unloaded["ok"], True)
eq("close: explicit unload releases VRAM", state["loaded"], None)

# Native load/unload must not depend on JIT being enabled.
state["loaded"], state["jit"] = None, False
out, seq = via_route("now")
eq("route: JIT-off still uses explicit load and unload", seq, ["load", "gen", "unload"])
ok("route: JIT-off generation succeeds", "subject_definitions" in out.get("result", ""), out)

srv.shutdown()
if fails:
    print(f"\n✗ {len(fails)} failed, {passed} passed\n")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print(f"✓ {passed} passed, 0 failed")
