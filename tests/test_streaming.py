"""Run: python3 tests/test_streaming.py"""
import importlib.util
import json
import pathlib
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PACK = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "h3stream", PACK / "__init__.py", submodule_search_locations=[str(PACK)])
module = importlib.util.module_from_spec(spec)
sys.modules["h3stream"] = module
spec.loader.exec_module(module)
L = __import__("h3stream.llm_backends", fromlist=["x"])

seen = []


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        seen.append(body)
        if body.get("model") == "legacy" and body.get("stream"):
            payload = json.dumps({"error": "unsupported field stream"}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if body.get("model") == "legacy":
            payload = json.dumps({"choices": [{"message": {"content":
                "subject_definitions: legacy\nsummary: fallback"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        events = [
            {"choices": [{"delta": {"content": "subject_definitions: x\n"}}]},
            {"choices": [{"delta": {"content": "summary: y\n"}}]},
            {"choices": [], "usage": {"prompt_tokens": 321, "completion_tokens": 24,
                                         "completion_tokens_details": {"reasoning_tokens": 0}}},
        ]
        for event in events:
            try:
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.flush()
                time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                return
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


server = ThreadingHTTPServer(("127.0.0.1", 3410), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()

chunks = []
text, metrics = L.stream_llm(
    "openai_compat", "http://127.0.0.1:3410/v1", "model", "", "",
    "system", "scene", on_delta=chunks.append)

checks = {
    "deltas arrive before final assembly": chunks == ["subject_definitions: x\n", "summary: y\n"],
    "final text is complete": text == "".join(chunks),
    "usage prompt tokens captured": metrics["prompt_tokens"] == 321,
    "usage completion tokens captured": metrics["completion_tokens"] == 24,
    "TTFT measured": isinstance(metrics["ttft_ms"], float) and metrics["ttft_ms"] >= 0,
    "default thinking is disabled": seen[0].get("chat_template_kwargs") == {"enable_thinking": False},
    "portable no-think token sent": seen[0]["messages"][1]["content"].endswith("/no_think"),
    "stream and usage requested": seen[0].get("stream") is True and "stream_options" in seen[0],
}

cancel = L.StreamCancel()
cancelled = False
try:
    L.stream_llm(
        "openai_compat", "http://127.0.0.1:3410/v1", "model", "", "",
        "system", "scene", on_delta=lambda _chunk: cancel.cancel(), cancel=cancel)
except L.LLMCancelled:
    cancelled = True
checks["cancellation closes an active upstream stream"] = cancelled

# A legacy server may reject `stream` itself. Its JSON fallback must retain the
# image part; a bare HTTP 400 used to trigger the unrelated media-shedding path.
legacy_chunks = []
legacy_text, legacy_metrics = L.stream_llm(
    "openai_compat", "http://127.0.0.1:3410/v1", "legacy", "", "",
    "system", "scene", images_base64=["QUJD"], on_delta=legacy_chunks.append)
legacy_requests = [body for body in seen if body.get("model") == "legacy"]
legacy_parts = legacy_requests[-1]["messages"][1]["content"]
checks["legacy non-stream fallback still answers"] = "subject_definitions" in legacy_text
checks["legacy fallback is marked non-streamed"] = legacy_metrics["streamed"] is False
checks["legacy stream rejection preserves the image"] = any(
    part.get("type") == "image_url" for part in legacy_parts)
checks["legacy fallback is emitted to the UI"] = legacy_chunks == [legacy_text]

server.shutdown()
failed = [name for name, ok in checks.items() if not ok]
for name, ok in checks.items():
    print(("  ✓ " if ok else "  ✗ ") + name)
sys.exit(1 if failed else 0)
