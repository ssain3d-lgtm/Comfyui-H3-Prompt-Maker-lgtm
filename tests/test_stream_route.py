"""Run: python3 tests/test_stream_route.py

End-to-end check of the aiohttp NDJSON bridge used by the ComfyUI overlay.
The backend unit test pins SSE parsing; this one pins the browser-facing event
order and makes sure usage metrics survive the thread/queue boundary.
"""
import asyncio
import importlib.util
import json
import pathlib
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from aiohttp import ClientSession, web

PACK = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "h3streamroute", PACK / "__init__.py", submodule_search_locations=[str(PACK)])
module = importlib.util.module_from_spec(spec)
sys.modules["h3streamroute"] = module
spec.loader.exec_module(module)
R = __import__("h3streamroute.server_routes", fromlist=["x"])


class LLM(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        events = [
            {"choices": [{"delta": {"content": "subject_definitions: a\n"}}]},
            {"choices": [{"delta": {"content": "summary: b\n"}}]},
            {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 5}},
        ]
        for event in events:
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.flush()
            time.sleep(0.02)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


stub = ThreadingHTTPServer(("127.0.0.1", 0), LLM)
threading.Thread(target=stub.serve_forever, daemon=True).start()


async def check():
    app = web.Application(client_max_size=64 * 1024 * 1024)
    routes = web.RouteTableDef()
    R.register(routes)
    app.add_routes(routes)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    body = {
        "promptText": "x",
        "llm": {
            "backend": "openai_compat",
            "base_url": f"http://127.0.0.1:{stub.server_port}/v1",
            "model": "m",
            "unload_after": "keep",
            "thinking": "off",
            "prompt_profile": "fast",
        },
    }
    try:
        async with ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{port}/h3_prompt_maker/api/generate-prompt",
                json=body,
                headers={"Accept": "application/x-ndjson"},
            ) as response:
                content_type = response.headers.get("Content-Type", "")
                events = [json.loads(line) async for line in response.content if line.strip()]
    finally:
        await runner.cleanup()

    assert content_type.startswith("application/x-ndjson"), content_type
    kinds = [event.get("type") for event in events]
    assert kinds[:2] == ["status", "status"], events
    assert "delta" in kinds, events
    assert kinds[-1] == "final", events
    final = events[-1]
    assert "subject_definitions" in final["result"], final
    assert final["metrics"]["prompt_tokens"] == 12, final
    assert final["metrics"]["completion_tokens"] == 5, final
    assert final["metrics"]["load_ms"] == 0.0, final


try:
    asyncio.run(check())
finally:
    stub.shutdown()
print("✓ aiohttp NDJSON route: event order, deltas, final result and usage metrics")
