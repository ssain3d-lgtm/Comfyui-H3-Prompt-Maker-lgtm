"""Run: python3 tests/test_thinking.py

What actually goes over the wire for each thinking mode — including a server
that rejects chat_template_kwargs, which is the whole reason for the fallback."""
import importlib.util, json, pathlib, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

P = pathlib.Path(__file__).resolve().parent.parent
sp = importlib.util.spec_from_file_location("h3p", P/"__init__.py", submodule_search_locations=[str(P)])
m = importlib.util.module_from_spec(sp); sys.modules["h3p"] = m; sp.loader.exec_module(m)
L = importlib.import_module("h3p.llm_backends")

seen, REJECT = [], {"on": False}
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        seen.append(body)
        if REJECT["on"] and "chat_template_kwargs" in body:
            b = json.dumps({"error": {"message": "Unrecognized request argument: chat_template_kwargs"}}).encode()
            self.send_response(400)
        else:
            b = json.dumps({"choices":[{"message":{"content":"subject_definitions: x\nsummary: y\ndetailed_description: z"}}]}).encode()
            self.send_response(200)
        self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b)))
        self.end_headers(); self.wfile.write(b)

srv = HTTPServer(("127.0.0.1", 3401), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()

passed, fails = 0, []
def ok(n, c, d=""):
    global passed
    if c: passed += 1
    else: fails.append(f"{n}{chr(10)+'      '+str(d) if d else ''}")

def call(mode, unload="keep", backend="openai_compat"):
    seen.clear()
    L.call_llm(backend, "http://127.0.0.1:3401/v1", "m", "", "", "sys", "장면 요청",
               thinking=mode, max_tokens=60000, unload_after=unload)
    return seen

b = call("auto")[0]
ok("auto: no template switch sent", "chat_template_kwargs" not in b)
ok("auto: user turn untouched", b["messages"][1]["content"] == "장면 요청", b["messages"][1]["content"])

b = call("off")[0]
ok("off: template switch says enable_thinking false",
   b.get("chat_template_kwargs") == {"enable_thinking": False}, b.get("chat_template_kwargs"))
ok("off: /no_think reaches the user turn", b["messages"][1]["content"].endswith("/no_think"))
ok("off: max_tokens still 60000", b["max_tokens"] == 60000, b["max_tokens"])

b = call("on")[0]
ok("on: template switch says enable_thinking true",
   b.get("chat_template_kwargs") == {"enable_thinking": True}, b.get("chat_template_kwargs"))
ok("on: /think reaches the user turn", b["messages"][1]["content"].endswith("/think"))

# The fallback: a server that has never heard of chat_template_kwargs
REJECT["on"] = True
calls = call("off")
ok("reject: retried without the unknown field", len(calls) == 2, f"{len(calls)} call(s)")
ok("reject: the retry dropped it", "chat_template_kwargs" not in calls[1])
ok("reject: /no_think still carries the intent", calls[1]["messages"][1]["content"].endswith("/no_think"))
REJECT["on"] = False
# --- what happens to the model after the answer -----------------------------
ok("keep: no time-to-live sent", L.unload_payload("keep", "lmstudio") == {})
ok("5m: LM Studio receives only ttl",
   L.unload_payload("5m", "lmstudio") == {"ttl": 300}, L.unload_payload("5m", "lmstudio"))
ok("now: LM Studio receives only the ttl compatibility fallback",
   L.unload_payload("now", "lmstudio") == {"ttl": 1}, L.unload_payload("now", "lmstudio"))
ok("5m: Ollama receives only keep_alive",
   L.unload_payload("5m", "ollama") == {"keep_alive": "5m"}, L.unload_payload("5m", "ollama"))
ok("now: Ollama receives only its immediate-unload spelling",
   L.unload_payload("now", "ollama") == {"keep_alive": 0}, L.unload_payload("now", "ollama"))
ok("unknown OpenAI-compatible servers get no backend-specific fields",
   L.unload_payload("now", "openai_compat") == {}, L.unload_payload("now", "openai_compat"))

b = call("auto", "now")[0]

# The model requested is the one the dialog saved — that is what makes the next
# generation reload the right thing rather than whatever was resident.
ok("reload: the saved model id is what gets requested", b["model"] == "m", b["model"])

# A backend that knows none of these optional fields must still generate.
REJECT["on"] = True
calls = call("off", "5m", "lmstudio")
ok("reject: one retry, not a failure", len(calls) == 2, f"{len(calls)} call(s)")
ok("reject: every optional field was shed",
   not any(k in calls[1] for k in ("chat_template_kwargs", "ttl", "keep_alive")),
   {k: calls[1][k] for k in ("chat_template_kwargs", "ttl", "keep_alive") if k in calls[1]})
ok("reject: the prompt itself still went", calls[1]["messages"][0]["content"] == "sys")
ok("reject: and max_tokens survived", calls[1]["max_tokens"] == 60000)
REJECT["on"] = False

srv.shutdown()

for f in fails: print("  ✗ " + f)
print(f"{'✓' if not fails else '✗'} {passed} passed, {len(fails)} failed")
sys.exit(1 if fails else 0)
