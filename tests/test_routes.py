#!/usr/bin/env python3
"""Tests for the overlay's HTTP layer that do not need aiohttp running.

These routes sit on ComfyUI's server, which has no authentication of its own,
so the asset handler is the one place in this pack where a caller-supplied
string reaches the filesystem. That guard is pinned here, along with the
request translation that decides what the LLM is actually asked.
"""

import importlib.util
import pathlib
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

# --- constants the frontend relies on ---------------------------------------
eq("prefix matches the app's API_BASE", R.PREFIX + "/api", "/h3_prompt_maker/api")
ok("body cap is generous enough for inline media", R.MAX_BODY_BYTES >= 32 * 1024 * 1024)

if failures:
    print(f"\n✗ {len(failures)} failed, {passed} passed\n")
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print(f"✓ all {passed} route assertions passed")
