"""Run: python3 tests/test_event_loop.py

These handlers are `async def` but everything they call — probe, warm-up, the
generation itself, model discovery — is synchronous urllib. Called directly they
hold the single event loop ComfyUI runs on for the entire request: measured at
3.00s against a 3s stub, and a local 14B model with a 60k budget holds it for
minutes. While it is held ComfyUI's websocket drops, the queue stops and /view
hangs, which reads to the user as ComfyUI crashing rather than a slow model.

A generation is allowed to be slow. Holding the host application hostage while
it runs is not, and nothing else in the suite would notice if it started again.
"""
import asyncio, importlib.util, json, pathlib, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
try:
    from aiohttp import web
except ModuleNotFoundError:  # pragma: no cover - environment problem, not a defect
    # Never skip: a guard that quietly opts out is a guard that stops guarding,
    # and this one is the only thing standing between a local generation and
    # ComfyUI freezing for its duration. Say what to install and fail.
    print("✗ aiohttp 가 필요합니다 — ComfyUI 가 쓰는 서버라 이 테스트도 씁니다.\n"
          "  python3 -m pip install aiohttp")
    raise SystemExit(1)

# Relative to this file, like every other test here. This started life as a
# throwaway script in a scratch directory and kept that directory's absolute
# path when it was promoted into tests/ — green locally, FileNotFoundError
# on any other machine, including CI.
PACK = pathlib.Path(__file__).resolve().parent.parent
sp = importlib.util.spec_from_file_location("h3l", PACK/"__init__.py", submodule_search_locations=[str(PACK)])
m = importlib.util.module_from_spec(sp); sys.modules["h3l"] = m; sp.loader.exec_module(m)
R = importlib.import_module("h3l.server_routes")

SLOW = 3.0
class LLM(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        b = json.dumps({"data":[{"id":"m"}]}).encode()
        self.send_response(200); self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length",0)))
        time.sleep(SLOW)
        b = json.dumps({"choices":[{"message":{"content":"subject_definitions: a\nsummary: b\ndetailed_description: c"}}]}).encode()
        self.send_response(200); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)
threading.Thread(target=HTTPServer(("127.0.0.1",3406),LLM).serve_forever, daemon=True).start()

worst = {"tick": 0.0, "health": 0.0}
async def heartbeat():
    while True:
        t = time.monotonic(); await asyncio.sleep(0.05)
        worst["tick"] = max(worst["tick"], time.monotonic()-t-0.05)

async def on_start(app): app["hb"] = asyncio.create_task(heartbeat())
app = web.Application(client_max_size=64*1024*1024)
app.on_startup.append(on_start)
routes = web.RouteTableDef(); R.register(routes); app.add_routes(routes)

def hammer():
    time.sleep(1.0)
    body = json.dumps({"promptText":"x","llm":{"backend":"openai_compat",
        "base_url":"http://127.0.0.1:3406/v1","model":"m","unload_after":"keep"}}).encode()
    threading.Thread(target=lambda: urllib.request.urlopen(
        urllib.request.Request("http://127.0.0.1:8813/h3_prompt_maker/api/generate-prompt",
        data=body, headers={"Content-Type":"application/json"}), timeout=60).read(), daemon=True).start()
    time.sleep(0.7)
    t = time.monotonic()
    urllib.request.urlopen("http://127.0.0.1:8813/h3_prompt_maker/api/health", timeout=60).read()
    worst["health"] = time.monotonic()-t
    time.sleep(SLOW+1.0)
    import os
    # Generous bounds: this asserts "the loop kept turning", not a latency SLA,
    # so it does not go red on a loaded CI runner. Before the fix both numbers
    # tracked the generation time exactly (3.00s and 2.31s here).
    fails = []
    if worst["tick"] > SLOW / 3:
        fails.append(f"event loop stalled {worst['tick']:.2f}s during a {SLOW}s generation")
    if worst["health"] > SLOW / 3:
        fails.append(f"/api/health waited {worst['health']:.2f}s behind a {SLOW}s generation")
    if fails:
        print("\n✗ " + "\n✗ ".join(fails))
        print("  블로킹 호출이 이벤트 루프로 돌아왔습니다 — _offthread 를 거치는지 확인하세요.")
        os._exit(1)
    print(f"✓ 느린 생성({SLOW}s) 중에도 루프는 계속 돎 "
          f"(최대 정지 {worst['tick']:.2f}s, /api/health {worst['health']:.2f}s)")
    os._exit(0)

threading.Thread(target=hammer, daemon=True).start()
web.run_app(app, host="127.0.0.1", port=8813, print=None)
