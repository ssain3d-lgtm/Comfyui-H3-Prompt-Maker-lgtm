"""Run: python3 tests/test_media_parts.py

What a clip and a sound file actually become on the wire.

The OpenAI-compatible chat schema has no video part at all, so a reference clip
travels as one contact sheet of its frames riding along as an image. Audio has a
part but almost no local model has an audio tower, so it is sent and shed. Both
of those are invisible failures if they go wrong — the model answers happily
having seen nothing — so the request bodies are pinned here.
"""
import base64
import importlib.util
import json
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

P = pathlib.Path(__file__).resolve().parent.parent
sp = importlib.util.spec_from_file_location("h3m", P / "__init__.py", submodule_search_locations=[str(P)])
m = importlib.util.module_from_spec(sp); sys.modules["h3m"] = m; sp.loader.exec_module(m)
L = importlib.import_module("h3m.llm_backends")

seen = []
REJECT = {"audio": False, "vision": False}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        seen.append(body)
        parts = body["messages"][1]["content"]
        kinds = {p.get("type") for p in parts} if isinstance(parts, list) else set()
        bad = ((REJECT["audio"] and "input_audio" in kinds)
               or (REJECT["vision"] and "image_url" in kinds))
        if bad:
            b = json.dumps({"error": {"message": "400: this model does not support that content type"}}).encode()
            self.send_response(400)
        else:
            b = json.dumps({"choices": [{"message": {"content":
                "subject_definitions: x\nsummary: y\ndetailed_description: z"}}]}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


srv = HTTPServer(("127.0.0.1", 3402), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()

passed, fails = 0, []


def ok(n, c, d=""):
    global passed
    if c:
        passed += 1
    else:
        fails.append(f"{n}{chr(10) + '      ' + str(d) if d else ''}")


def eq(n, a, b):
    ok(n, a == b, f"expected {b!r}, got {a!r}")


JPEG = base64.b64encode(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01").decode()
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR").decode()
WAV = base64.b64encode(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00").decode()
MP3 = base64.b64encode(b"ID3\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00").decode()
OGG = base64.b64encode(b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00").decode()
M4A = base64.b64encode(b"\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00").decode()


def call(images=None, audios=None, thinking="auto", unload="keep", backend="openai_compat"):
    seen.clear()
    return L.call_llm(backend, "http://127.0.0.1:3402/v1", "m", "", "", "sys", "장면 요청",
                      images_base64=images, audios_base64=audios, max_tokens=60000,
                      thinking=thinking, unload_after=unload)


def parts(body):
    c = body["messages"][1]["content"]
    return c if isinstance(c, list) else []


# --- container sniffing -----------------------------------------------------
# A wav payload handed over as "mp3" is decoded as noise and then described as
# if it were the clip, which is worse than not sending it.
eq("format: RIFF/WAVE reads as wav", L.audio_format(WAV), "wav")
eq("format: an ID3 header reads as mp3", L.audio_format(MP3), "mp3")
eq("format: OggS reads as ogg", L.audio_format(OGG), "ogg")
eq("format: an ftyp box reads as m4a", L.audio_format(M4A), "m4a")
eq("format: a frame-sync mp3 with no ID3 still reads as mp3",
   L.audio_format(base64.b64encode(b"\xff\xfb\x90\x64" + b"\x00" * 12).decode()), "mp3")
eq("format: garbage falls back rather than raising", L.audio_format("!!!!"), "wav")
eq("format: an empty payload falls back rather than raising", L.audio_format(""), "wav")
# webp and wav share the RIFF magic — the four bytes at offset 8 are the only
# thing that separates them, and getting it wrong sends a picture as audio.
eq("format: a RIFF/WEBP payload is NOT taken for wav",
   L.audio_format(base64.b64encode(b"RIFF\x24\x00\x00\x00WEBPVP8 ").decode()), "wav")
eq("mime: the same webp payload is recognised as an image",
   L.image_mime(base64.b64encode(b"RIFF\x24\x00\x00\x00WEBPVP8 ").decode()), "image/webp")

# --- the wire format --------------------------------------------------------
REJECT["audio"] = REJECT["vision"] = False
call(images=[JPEG, PNG])
p = parts(seen[0])
eq("images: one text part then one part per picture", [x["type"] for x in p],
   ["text", "image_url", "image_url"])
ok("images: a jpeg contact sheet is labelled image/jpeg",
   p[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"), p[1]["image_url"]["url"][:40])
ok("images: a png reference photo is still image/png",
   p[2]["image_url"]["url"].startswith("data:image/png;base64,"), p[2]["image_url"]["url"][:40])
eq("images: the payload survives the round trip",
   p[1]["image_url"]["url"].split(",", 1)[1], JPEG)

call(images=[JPEG], audios=[WAV, MP3])
p = parts(seen[0])
eq("audio: pictures come first, audio after", [x["type"] for x in p],
   ["text", "image_url", "input_audio", "input_audio"])
eq("audio: the format travels with the payload", p[2]["input_audio"]["format"], "wav")
eq("audio: each clip gets its own format", p[3]["input_audio"]["format"], "mp3")
eq("audio: the payload is bare base64, not a data URL", p[2]["input_audio"]["data"], WAV)
eq("audio: exactly one request when nothing is rejected", len(seen), 1)

# --- shedding ---------------------------------------------------------------
# The common case: a text or vision model that has no audio tower. Losing the
# pictures at the same time would throw away a reference it could have read.
REJECT["audio"], REJECT["vision"] = True, False
call(images=[JPEG], audios=[WAV])
eq("shed: exactly one retry after an audio rejection", len(seen), 2)
eq("shed: the retry drops audio", [x["type"] for x in parts(seen[1])], ["text", "image_url"])
ok("shed: the retry keeps the picture", parts(seen[1])[1]["image_url"]["url"].endswith(JPEG))
ok("shed: the prompt survives", parts(seen[1])[0]["text"] == "장면 요청")
eq("shed: max_tokens survives the retry", seen[1]["max_tokens"], 60000)

# The overlay defaults to immediate unload, and a user may also turn thinking
# off. Both add optional request fields. The first 400 used to shed those fields
# and return directly from retry 1; retry 2 hit the same unsupported audio and
# escaped without ever reaching the audio-removal branch.
REJECT["audio"], REJECT["vision"] = True, False
call(images=[JPEG], audios=[WAV], thinking="off", unload="5m", backend="lmstudio")
eq("shed+optional: optional fields then audio are handled in one retry loop", len(seen), 3)
ok("shed+optional: retry 2 removed template and ttl",
   not any(k in seen[1] for k in ("chat_template_kwargs", "ttl", "keep_alive")), seen[1])
eq("shed+optional: final request keeps the picture and drops audio",
   [x["type"] for x in parts(seen[2])], ["text", "image_url"])

# A text-only model rejects both. Audio goes first, then everything.
REJECT["audio"] = REJECT["vision"] = True
call(images=[JPEG], audios=[WAV])
eq("shed: a text-only model costs two retries, not more", len(seen), 3)
eq("shed: the last attempt is plain text", seen[2]["messages"][1]["content"], "장면 요청")

# Nothing to shed means nothing to retry — a dead server must not be hit twice.
REJECT["audio"], REJECT["vision"] = False, True
try:
    call(images=None, audios=None)
except Exception:
    pass
eq("shed: a text-only request is never retried", len(seen), 1)

srv.shutdown()
if fails:
    print(f"\n✗ {len(fails)} failed, {passed} passed\n")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print(f"✓ {passed} passed, 0 failed")
