#!/usr/bin/env python3
"""Tests for the overlay's HTTP layer that do not need aiohttp running.

These routes sit on ComfyUI's server, which has no authentication of its own,
so the asset handler is the one place in this pack where a caller-supplied
string reaches the filesystem. That guard is pinned here, along with the
request translation that decides what the LLM is actually asked.
"""

import importlib.util
import json
import pathlib
import re
import sys

# server_routes uses package-relative imports, as a ComfyUI custom node must.
# The pack directory name has hyphens, so it is loaded by path rather than by
# name — the same way ComfyUI itself loads custom_nodes.
_PACK = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "h3pack", _PACK / "__init__.py", submodule_search_locations=[str(_PACK)])
_pack = importlib.util.module_from_spec(_spec)
sys.modules["h3pack"] = _pack
_spec.loader.exec_module(_pack)
R = importlib.import_module("h3pack.server_routes")

passed, failures = 0, []


def eq(name, actual, expected):
    global passed
    if actual == expected:
        passed += 1
    else:
        failures.append(f"{name}\n      expected: {expected!r}\n      actual:   {actual!r}")


def ok(name, cond, detail=""):
    global passed
    if cond:
        passed += 1
    else:
        failures.append(f"{name}{chr(10) + '      ' + detail if detail else ''}")


# --- asset path guard -------------------------------------------------------
ok("asset: index.html resolves", R._safe_asset("index.html") is not None
   or not (R.APP_DIR / "index.html").is_file())
ok("asset: traversal with ../ is refused", R._safe_asset("../../nodes.py") is None)
ok("asset: absolute path is refused", R._safe_asset("/etc/passwd") is None)
ok("asset: encoded traversal is refused", R._safe_asset("assets/../../../nodes.py") is None)
ok("asset: python file inside the dir is refused", R._safe_asset("../server_routes.py") is None)
ok("asset: unknown extension is refused", R._safe_asset("index.exe") is None)
ok("asset: missing file is refused", R._safe_asset("nope.js") is None)

# --- data URL handling ------------------------------------------------------
eq("dataurl: strips the prefix", R._strip_data_url("data:image/png;base64,AAAB"), "AAAB")
eq("dataurl: passes a bare payload", R._strip_data_url("AAAB"), "AAAB")
eq("dataurl: non-string is dropped", R._strip_data_url(None), None)
eq("dataurl: comma-free data URL is not truncated", R._strip_data_url("data:image/png;base64"), "data:image/png;base64")

# --- collecting the singular and plural media fields ------------------------
eq("collect: merges single and list", R._collect({"a": "x", "b": ["y", "z"]}, "a", "b"), ["x", "y", "z"])
eq("collect: skips empties and wrong types", R._collect({"a": "", "b": [1, "y", None]}, "a", "b"), ["y"])
eq("collect: missing keys are fine", R._collect({}, "a", "b"), [])

# --- llm settings -----------------------------------------------------------
d = R._llm_settings({})
eq("llm: defaults to lmstudio", d["backend"], "lmstudio")
eq("llm: default temperature", d["temperature"], 0.7)
eq("llm: defaults to unload-on-close", d["unload_after"], "close")
eq("llm: an explicit keep is preserved",
   R._llm_settings({"llm": {"unload_after": "keep"}})["unload_after"], "keep")
eq("llm: an invalid unload mode falls back to close",
   R._llm_settings({"llm": {"unload_after": "later"}})["unload_after"], "close")
eq("llm: defaults to the compact prompt", d["prompt_profile"], "fast")
eq("llm: full prompt remains selectable",
   R._llm_settings({"llm": {"prompt_profile": "full"}})["prompt_profile"], "full")
eq("llm: an invalid prompt profile falls back safely",
   R._llm_settings({"llm": {"prompt_profile": "huge"}})["prompt_profile"], "fast")
eq("llm: unknown backend falls back", R._llm_settings({"llm": {"backend": "hax"}})["backend"], "lmstudio")
eq("llm: v1 alias still resolves", R._llm_settings({"llm": {"backend": "openai_compatible"}})["backend"], "openai_compat")
eq("llm: temperature is clamped high", R._llm_settings({"llm": {"temperature": 99}})["temperature"], 2.0)
eq("llm: temperature is clamped low", R._llm_settings({"llm": {"temperature": -5}})["temperature"], 0.0)
eq("llm: junk temperature falls back", R._llm_settings({"llm": {"temperature": "hot"}})["temperature"], 0.7)
eq("llm: non-dict llm block is ignored", R._llm_settings({"llm": "nope"})["backend"], "lmstudio")

# --- user text assembly -----------------------------------------------------
body = {
    "minimaxStyle": "ref2va",
    "promptText": "여자가 골목을 걷는다",
    "ltxNarration": "안녕",
    "voiceDirection": "낮고 허스키한 30대 목소리",
    "imageRoles": ["얼굴 기준", ""],
    "videoRefNote": "걸음걸이만",
    "audioRefNote": "음색만",
}
text = R._build_user_text(body, 2)
ok("user: names the submode", "[MINIMAX H3 REF2VA REQUEST]" in text, text[:80])
ok("user: carries the scene", "여자가 골목을 걷는다" in text)
ok("user: tags each supplied picture", "<Picture 1> — 얼굴 기준" in text and "<Picture 2>" in text, text)
ok("user: an empty role gets no dash", "<Picture 2> —" not in text, text)
ok("user: voice goes into the subject, not a bare order", "<Subject N>" in text)
# The app's own hard rule: H3 is CFG 1, so a reference note must never become a ban
ok("user: reference notes are steered to slots, not prohibitions",
   "not as a prohibition" in text and "Do not copy" not in text, text)
ok("user: no picture line when nothing is attached",
   "Reference pictures supplied" not in R._build_user_text(body, 0))

remake = R._build_user_text({"isRemake": True, "remakeSourcePrompt": "old prompt here"}, 0)
ok("user: remake source is included", "[REMAKE SOURCE PROMPT]" in remake and "old prompt here" in remake)
ok("user: no remake block without a source",
   "[REMAKE SOURCE PROMPT]" not in R._build_user_text({"isRemake": True}, 0))

# --- contact sheets: the only way a clip reaches a local model ---------------
# The OpenAI-compatible chat schema has no video part, so before this an
# attached clip reached the model as nothing but the note beside it. What
# travels now is one labelled sheet of its frames, riding along as an image.
sheet_text = R._build_user_text(body, 2, 1)
ok("sheet: the model is told the last image is not a still",
   "not stills" in sheet_text or "contact sheet" in sheet_text, sheet_text[-300:])
ok("sheet: it is told how many frames and in what order",
   "left to right and top to bottom" in sheet_text, sheet_text[-300:])
ok("sheet: it is told to read motion, not describe a grid",
   "must not appear in the prompt" in sheet_text, sheet_text[-300:])
ok("sheet: the frame count from the request is quoted",
   "8 frames" in R._build_user_text(dict(body, videoFrameCount=8), 2, 1))
eq("sheet: a different frame count is quoted, not the default",
   "16 frames" in R._build_user_text(dict(body, videoFrameCount=16), 2, 1), True)
ok("sheet: no sheet block when no clip was attached",
   "contact sheet" not in R._build_user_text(body, 2, 0))
eq("sheet: one clip gets one tag, not a range from 1 to 1",
   "<Video 1> ... <Video 1>" in sheet_text, False)
ok("sheet: one clip is still tagged", "<Video 1>" in sheet_text)
ok("sheet: three clips get a range",
   "<Video 1> ... <Video 3>" in R._build_user_text(body, 2, 3))

ok("sheet: pictures are still described alongside",
   "<Picture 1> — 얼굴 기준" in sheet_text)

# Where the sheets land in the image list decides whether a third clip can push
# <Picture 9> out of the request. They go after the pictures, with their own cap.
_sheet_body = {
    "imagesBase64": ["P%d" % i for i in range(12)],
    "videoFramesBase64": ["S1", "S2", "S3", "S4"],
}
_imgs = [R._strip_data_url(x) for x in R._collect(_sheet_body, "imageBase64", "imagesBase64")][:9]
_sheets = [R._strip_data_url(x) for x in R._collect(_sheet_body, "videoFramesBase64")][:3]
eq("sheet: pictures keep all nine of their slots", _imgs, ["P%d" % i for i in range(9)])
eq("sheet: sheets are capped at three, like the video slots", _sheets, ["S1", "S2", "S3"])
eq("sheet: sheets follow the pictures, never displace them",
   (_imgs + _sheets)[:9], ["P%d" % i for i in range(9)])
eq("sheet: a data URL prefix is stripped like any other attachment",
   R._strip_data_url("data:image/jpeg;base64,AAAA"), "AAAA")

# A JPEG sheet labelled image/png works on servers that sniff the bytes and
# fails on the strict ones — the worst kind of bug to chase.
B = importlib.import_module("h3pack.llm_backends")
eq("sheet: a jpeg payload is labelled image/jpeg", B.image_mime("/9j/4AAQSkZJRg"), "image/jpeg")
eq("sheet: a png payload is still image/png", B.image_mime("iVBORw0KGgoAAAA"), "image/png")
eq("sheet: webp is recognised", B.image_mime("UklGRiQAAABXRUJQ"), "image/webp")
eq("sheet: an unknown payload falls back to png rather than failing", B.image_mime("zzzz"), "image/png")
eq("sheet: a non-string never raises", B.image_mime(None), "image/png")

# --- the trailing slash the relative asset links depend on ------------------
_html = (R.APP_DIR / "index.html")
if _html.is_file():
    _src = _html.read_text(encoding="utf-8")
    ok("bundle: assets are linked relatively", './assets/' in _src, _src[:200])
    ok("bundle: no absolute asset path would escape the mount point",
       'src="/assets' not in _src and 'href="/index.css"' not in _src)
_js = (_PACK / "web" / "h3_maker.js").read_text(encoding="utf-8")
ok("iframe src keeps the trailing slash — without it every ./asset resolves one level too high",
   "${PREFIX}/app/`" in _js, [l for l in _js.splitlines() if "frame.src" in l])
ok("bare /app redirects instead of serving", "HTTPMovedPermanently" in
   (_PACK / "server_routes.py").read_text(encoding="utf-8"))

# --- reading the request body -----------------------------------------------
# A body carrying a base64 image does not arrive in one piece. aiohttp's
# StreamReader.read(n) hands back only what is buffered, so reading that way
# truncated the JSON and every generation with an attachment came back
# "잘못된 요청입니다: Unterminated string". These fakes reproduce that split.
import asyncio


class _FakeStream:
    """Mimics aiohttp's StreamReader for the two ways of draining it."""

    def __init__(self, payload, chunk=64):
        self._pieces = [payload[i:i + chunk] for i in range(0, len(payload), chunk)] or [b""]

    def iter_chunked(self, _n):
        pieces = list(self._pieces)

        class _It:
            def __aiter__(self_inner):
                return self_inner

            async def __anext__(self_inner):
                if not pieces:
                    raise StopAsyncIteration
                return pieces.pop(0)

        return _It()

    async def read(self, _n=-1):
        # The behaviour that caused the bug: one buffered piece, not the lot.
        return self._pieces[0]


class _FakeRequest:
    def __init__(self, payload, chunk=64):
        self.content = _FakeStream(payload, chunk)


_payload = json.dumps({
    "promptText": "골목을 걷는 인물",
    "imagesBase64": ["data:image/png;base64," + "A" * 5000],
}).encode()

_whole = asyncio.new_event_loop().run_until_complete(R._read_body(_FakeRequest(_payload)))
eq("body: reads every chunk, not just the buffered one", len(_whole), len(_payload))
ok("body: the JSON parses after a chunked read", json.loads(_whole.decode())["promptText"] == "골목을 걷는 인물")
ok("body: a single-chunk read would have truncated it — the bug is real",
   len(asyncio.new_event_loop().run_until_complete(_FakeRequest(_payload).content.read(len(_payload)))) < len(_payload))

_small = asyncio.new_event_loop().run_until_complete(
    R._read_body(_FakeRequest(b'{"a":1}'), limit=R.MAX_BODY_BYTES))
eq("body: a tiny body still round-trips", json.loads(_small.decode()), {"a": 1})

_over = asyncio.new_event_loop().run_until_complete(R._read_body(_FakeRequest(b"x" * 5000), limit=1000))
eq("body: over the cap returns None rather than a partial parse", _over, None)

_empty = asyncio.new_event_loop().run_until_complete(R._read_body(_FakeRequest(b"")))
eq("body: an empty body is empty, not an error", _empty, b"")

# --- max_tokens, typed in the dialog and carried to the model ---------------
# A reasoning model spends this budget thinking. At the old 8192 default a
# Qwen3-class model burned it inside <think> and answered with one line.
eq("tokens: default is generous enough for a thinking model",
   R._llm_settings({})["max_tokens"], 60000)
eq("tokens: a typed value is honoured", R._llm_settings({"llm": {"max_tokens": 32000}})["max_tokens"], 32000)
eq("tokens: a string from the widget still parses",
   R._llm_settings({"llm": {"max_tokens": "120000"}})["max_tokens"], 120000)
eq("tokens: junk falls back rather than sending NaN",
   R._llm_settings({"llm": {"max_tokens": "lots"}})["max_tokens"], 60000)
eq("tokens: absurdly small is floored", R._llm_settings({"llm": {"max_tokens": 10}})["max_tokens"], 1024)
eq("tokens: absurdly large is capped", R._llm_settings({"llm": {"max_tokens": 99999999}})["max_tokens"], 1000000)

_src = (_PACK / "server_routes.py").read_text(encoding="utf-8")
ok("tokens: the route actually forwards it to call_llm", "max_tokens=cfg[\"max_tokens\"]" in _src)
_js = (_PACK / "web" / "h3_maker.js").read_text(encoding="utf-8")
ok("tokens: the dialog offers a field for it", "inputs.max_tokens" in _js)
ok("tokens: and saves what was typed", 'max_tokens: Number(inputs.max_tokens.value)' in _js)

# --- thinking switch --------------------------------------------------------
from h3pack.llm_backends import apply_thinking  # noqa: E402

eq("thinking: defaults off for local Qwen speed", R._llm_settings({})["thinking"], "off")
eq("thinking: off is honoured", R._llm_settings({"llm": {"thinking": "off"}})["thinking"], "off")
eq("thinking: on is honoured", R._llm_settings({"llm": {"thinking": "on"}})["thinking"], "on")
eq("thinking: an unknown mode falls back rather than reaching the model",
   R._llm_settings({"llm": {"thinking": "sometimes"}})["thinking"], "off")

eq("thinking: auto leaves the user turn untouched", apply_thinking("장면", "auto"), "장면")
ok("thinking: off appends the token Qwen3 reads", apply_thinking("장면", "off").endswith("/no_think"))
ok("thinking: on appends the opposite", apply_thinking("장면", "on").endswith("/think"))
ok("thinking: the scene text survives either way", apply_thinking("장면", "off").startswith("장면"))

_src = (_PACK / "server_routes.py").read_text(encoding="utf-8")
ok("thinking: the route forwards it", 'thinking=cfg["thinking"]' in _src)
_js = (_PACK / "web" / "h3_maker.js").read_text(encoding="utf-8")
ok("thinking: the dialog offers the three modes",
   'inputs.thinking' in _js and '"off"' in _js and '"on"' in _js)
ok("thinking: and saves the choice", 'thinking: inputs.thinking.value' in _js)

# --- one repository name, everywhere ----------------------------------------
# The repo was reachable under two spellings for a while (GitHub redirects a
# renamed repo), so the README, pyproject and the web app's prompts.ts drifted
# apart. A redirect is not forever: it breaks the day someone registers the old
# name. The clone URL in the README is also the folder name a user ends up with
# under custom_nodes, so it is the one that has to be right.
REPO_NAME = "Comfyui-H3-Prompt-Maker-lgtm"
for _rel in ("README.md", "pyproject.toml"):
    _text = (_PACK / _rel).read_text(encoding="utf-8")
    _urls = re.findall(r"github\.com/ssain3d-lgtm/([A-Za-z0-9._-]+?)(?:\.git)?(?=[\s\)\"'<]|$)", _text)
    # Only the URLs that point at THIS pack — the README also links the web app
    # repo the overlay and prompts are extracted from, which is a different one.
    _mine = {u for u in _urls if "lgtm" in u.lower() and "h3-prompt-maker" in u.lower()}
    ok(f"repo: {_rel} links this pack at all", bool(_mine), f"found {_urls}")
    for _name in _mine:
        eq(f"repo: {_rel} spells it {REPO_NAME}", _name, REPO_NAME)

# --- constants the frontend relies on ---------------------------------------
eq("prefix matches the app's API_BASE", R.PREFIX + "/api", "/h3_prompt_maker/api")
ok("body cap is generous enough for inline media", R.MAX_BODY_BYTES >= 32 * 1024 * 1024)

if failures:
    print(f"\n✗ {len(failures)} failed, {passed} passed\n")
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print(f"✓ all {passed} route assertions passed")
