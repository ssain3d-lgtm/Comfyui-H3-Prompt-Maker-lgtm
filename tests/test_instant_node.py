#!/usr/bin/env python3
"""The UI-Instant node: the overlay's saved form, generated on Queue.

The UI node deliberately never calls the model — Queue emits what was applied.
This node is the opposite contract, so what is pinned here is that it asks the
model for the same thing the overlay would have (same request translation),
takes its pictures from the socket, lets a wired scene override the form, and
fails loudly instead of silently when there is nothing to generate from.
"""

import importlib.util
import json
import pathlib
import sys

_PACK = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "h3pack", _PACK / "__init__.py", submodule_search_locations=[str(_PACK)])
_pack = importlib.util.module_from_spec(_spec)
sys.modules["h3pack"] = _pack
_spec.loader.exec_module(_pack)
N = importlib.import_module("h3pack.nodes")
R = importlib.import_module("h3pack.server_routes")

passed, failures = 0, []


def ok(name, cond, detail=""):
    global passed
    if cond:
        passed += 1
    else:
        failures.append(f"{name}{chr(10) + '      ' + detail if detail else ''}")


def eq(name, actual, expected):
    ok(name, actual == expected, f"expected {expected!r}, got {actual!r}")


ANSWER = "```\nintegrated_multimodal_description: [Shot 1] she walks.\n\noverall_soundscape: rain.\n\nnon_diegetic_music: N/A\n```\nlength 243 (10.13 s)"

calls = []
warmed = []


def fake_call_llm(backend, base_url, model, api_key, cli_command, system_prompt, user_text, **kw):
    calls.append({"backend": backend, "model": model, "system": system_prompt, "user": user_text, **kw})
    return ANSWER


def fake_warm(backend, base_url, api_key, model, timeout=180.0):
    warmed.append(model)
    return {"ok": True, "detail": "stub"}


N.call_llm = fake_call_llm
N.warm_up_model = fake_warm

node = N.H3PromptMakerInstant()
FORM = {
    "prompt": "여자가 골목을 걷는다", "ltxNarration": "안녕", "voiceDirection": "낮은 목소리",
    "duration": 10, "isNSFW": False, "minimaxStyle": "i2va",
    "cameraPosition": "low angle", "cameraAngle": "", "customSystemPrompt": "keep it moody",
    "isRemakeMode": False, "remakeSourcePrompt": "", "remakeAxes": [], "remakeStrength": "medium",
    "remakeSourceType": "h3", "videoNote": "걸음걸이만", "audioNote": "",
}
LLM = {"settings_version": 2, "backend": "lmstudio", "model": "qwen3-14b", "server_model": "(auto)",
       "temperature": 0.7, "max_tokens": 60000, "thinking": "off", "prompt_profile": "fast",
       "unload_after": "close"}

# --- registration -------------------------------------------------------------
ok("registered under its own id", "H3PromptMakerInstant" in N.NODE_CLASS_MAPPINGS)
ok("display name says what it is",
   "UI-Instant" in N.NODE_DISPLAY_NAME_MAPPINGS["H3PromptMakerInstant"])
inputs = N.H3PromptMakerInstant.INPUT_TYPES()
eq("same hidden widgets as the UI node, so the overlay and ⚙️ dialog work unchanged",
   sorted(k for k in inputs["required"] if k != "seed"), ["llm", "result", "state"])
ok("pictures come from a socket", inputs["optional"]["images"] == ("IMAGE",))
ok("scene_request is a socket, not a widget", inputs["optional"]["scene_request"][1].get("forceInput") is True)
eq("same outputs as the UI node", N.H3PromptMakerInstant.RETURN_NAMES, N.H3PromptMakerUI.RETURN_NAMES)

# --- the happy path -------------------------------------------------------------
out = node.generate(state=json.dumps(FORM), llm=json.dumps(LLM), seed=7)
eq("one model call per Queue", len(calls), 1)
c = calls[0]
eq("the prompt socket carries the parsed body", out[0], "integrated_multimodal_description: [Shot 1] she walks.\n\noverall_soundscape: rain.\n\nnon_diegetic_music: N/A")
eq("frames come from the length line", out[1], 243)
eq("segment count", out[4], 1)
ok("the form's scene reaches the model", "여자가 골목을 걷는다" in c["user"], c["user"][:200])
ok("the form's submode reaches the model", "[MINIMAX H3 I2VA REQUEST]" in c["user"], c["user"][:80])
ok("dialogue and voice reach the model", "안녕" in c["user"] and "낮은 목소리" in c["user"])
ok("the video note reaches the model", "걸음걸이만" in c["user"])
ok("camera reaches the system prompt", "low angle" in c["system"])
ok("custom directives reach the system prompt", "keep it moody" in c["system"])
ok("SFW form gets the SFW preamble", "BLOCK_NONE" not in c["system"] and c["system"] == R.prepare_generation(
    N.instant_request_body(FORM, LLM, FORM["prompt"], []))["system_prompt"])
eq("backend settings travel from the llm widget", (c["backend"], c["model"]), ("lmstudio", "qwen3-14b"))
eq("thinking / max_tokens / profile travel too", (c["thinking"], c["max_tokens"]), ("off", 60000))
eq("the seed is passed through", c["seed"], 7)
eq("'창 닫을 때 언로드' becomes 'unload now' — there is no window to close in a graph",
   c["unload_after"], "now")
eq("a model that unloads is warmed before the call", warmed, ["qwen3-14b"])

# --- the request is the overlay's request ---------------------------------------
body = N.instant_request_body(FORM, LLM, FORM["prompt"], ["QUJD"])
prep = R.prepare_generation(body)
eq("pictures from the socket are labelled like attachments", prep["send_images"], ["QUJD"])
ok("and announced in the user turn", "<Picture 1>" in prep["user_text"], prep["user_text"])
eq("the llm block is the saved settings", body["llm"], LLM)

# --- overrides and modes ---------------------------------------------------------
calls.clear(); warmed.clear()
node.generate(state=json.dumps(FORM), llm=json.dumps(LLM), scene_request="  a man runs  ")
ok("a wired scene_request replaces the form's scene",
   "a man runs" in calls[0]["user"] and "여자가 골목을" not in calls[0]["user"], calls[0]["user"][:200])

calls.clear(); warmed.clear()
node.generate(state=json.dumps({**FORM, "isNSFW": True}), llm=json.dumps({**LLM, "unload_after": "keep"}))
ok("NSFW form gets the NSFW system prompt",
   calls[0]["system"] != c["system"] and calls[0]["system"] == R.prepare_generation(
       N.instant_request_body({**FORM, "isNSFW": True}, {**LLM, "unload_after": "keep"}, FORM["prompt"], []))["system_prompt"])
eq("keep stays keep", calls[0]["unload_after"], "keep")
eq("and nothing is warmed when the model stays resident", warmed, [])

calls.clear(); warmed.clear()
node.generate(state=json.dumps(FORM), llm=json.dumps({**LLM, "backend": "gemini", "model": "gemini-3.5-flash"}))
eq("gemini: warm-up is skipped by warm_up_model itself, the call goes through", len(calls), 1)

calls.clear()
remake_form = {**FORM, "prompt": "", "isRemakeMode": True, "remakeSourcePrompt": "old prompt here",
               "remakeAxes": ["camera", 5], "remakeStrength": "reimagine", "remakeSourceType": "custom"}
node.generate(state=json.dumps(remake_form), llm=json.dumps(LLM))
ok("remake mode with no scene still generates, from the source prompt",
   "[REMAKE SOURCE PROMPT]" in calls[0]["user"] and "old prompt here" in calls[0]["user"])
ok("the remake directive is in the system prompt", "REMAKE" in calls[0]["system"])

# --- failing loudly ---------------------------------------------------------------
def raises(fn, needle):
    try:
        fn()
    except RuntimeError as exc:
        return needle in str(exc)
    return False

ok("nothing to generate from is an error naming the three ways to fix it",
   raises(lambda: node.generate(state="{}", llm=json.dumps(LLM)), "scene_request"))
ok("no model settings is an error pointing at ⚙️",
   raises(lambda: node.generate(state=json.dumps(FORM), llm=""), "모델 연결"))
ok("garbage widgets do not crash the node", raises(lambda: node.generate(state="{not json", llm="[1,2]"), "장면"))


def failing(*a, **k):
    raise N.LLMError("connection refused")

N.call_llm = failing
ok("an LLM error surfaces as a node error, not a traceback",
   raises(lambda: node.generate(state=json.dumps(FORM), llm=json.dumps(LLM)), "connection refused"))
N.call_llm = lambda *a, **k: "   "
ok("an empty answer is an error", raises(lambda: node.generate(state=json.dumps(FORM), llm=json.dumps(LLM)), "빈 응답"))

# --- the UI node is untouched -------------------------------------------------------
N.call_llm = fake_call_llm
calls.clear()
ui = N.H3PromptMakerUI()
ui.emit(state=json.dumps(FORM), llm=json.dumps(LLM), result=json.dumps({"prompt": "applied", "lengthFrames": 158}))
eq("the UI node still never calls the model on Queue", calls, [])

if failures:
    print(f"\n✗ {len(failures)} failed, {passed} passed\n")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"✓ all {passed} UI-Instant node assertions passed")
