#!/usr/bin/env python3
"""Freeing ComfyUI's VRAM before a local LLM generation.

Two things make this worth pinning. It touches somebody's render — unloading
a model out from under a running sampler is the one way this can do damage, so
the busy check is load-bearing. And it runs on every generation, so a failure
here must never take the generation down with it.

ComfyUI is not importable in CI, so `comfy.model_management` and `server` are
stubbed into sys.modules and the real code path runs against them.
"""

import importlib.util
import pathlib
import sys
import types

_PACK = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "h3pack", _PACK / "__init__.py", submodule_search_locations=[str(_PACK)])
_pack = importlib.util.module_from_spec(_spec)
sys.modules["h3pack"] = _pack
_spec.loader.exec_module(_pack)
CM = importlib.import_module("h3pack.comfy_memory")
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


# --- a fake ComfyUI ----------------------------------------------------------
class FakeMM:
    def __init__(self, free=2 * 1024 ** 3, after_unload=9 * 1024 ** 3, old_signature=False):
        self.calls = []
        self.free = free
        self.after_unload = after_unload
        self.old_signature = old_signature

    def get_torch_device(self):
        return "cuda:0"

    def get_free_memory(self, dev):
        return self.free

    def unload_all_models(self):
        self.calls.append("unload_all_models")
        self.free = self.after_unload

    def soft_empty_cache(self, force=False):
        if self.old_signature and force is not False:
            raise TypeError("soft_empty_cache() takes 0 positional arguments")
        self.calls.append(f"soft_empty_cache({force})")


class FakeQueue:
    def __init__(self, running=(), raises=False):
        self.flags = {}
        self.running = list(running)
        self.raises = raises

    def set_flag(self, name, data):
        self.flags[name] = data

    def get_current_queue(self):
        if self.raises:
            raise RuntimeError("no queue")
        return self.running, []


def install(mm=None, queue=None):
    """Put a fake ComfyUI on sys.modules, or remove it when both are None."""
    for name in ("comfy", "comfy.model_management", "server"):
        sys.modules.pop(name, None)
    if mm is not None:
        pkg = types.ModuleType("comfy")
        mod = types.ModuleType("comfy.model_management")
        for attr in ("get_torch_device", "get_free_memory", "unload_all_models", "soft_empty_cache"):
            setattr(mod, attr, getattr(mm, attr))
        pkg.model_management = mod
        sys.modules["comfy"] = pkg
        sys.modules["comfy.model_management"] = mod
    if queue is not None:
        srv = types.ModuleType("server")
        srv.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(prompt_queue=queue))
        sys.modules["server"] = srv


# --- the modes ----------------------------------------------------------------
eq("the mode list is what the dialog offers", CM.FREE_MODES, ["off", "cache", "models"])
eq("freeing models is the default", CM.DEFAULT_FREE_MODE, "models")

install(None, None)
r = CM.free_comfy_memory("off")
ok("off does nothing at all", r["ran"] is False and "꺼져" in r["detail"], str(r))
r = CM.free_comfy_memory("models")
ok("outside ComfyUI it is a no-op, not an error", r["ran"] is False and "ComfyUI 밖" in r["detail"], str(r))
eq("an unknown mode is treated as off, never as 'unload everything'",
   CM.free_comfy_memory("wipe-it-all")["ran"], False)
eq("describe says nothing when nothing ran", CM.describe(r), "")

# --- the idle case, which is the normal one ------------------------------------
mm, q = FakeMM(), FakeQueue()
install(mm, q)
r = CM.free_comfy_memory("models")
ok("models: every loaded model is unloaded", "unload_all_models" in mm.calls, str(mm.calls))
ok("models: the allocator's blocks are released after", mm.calls[-1].startswith("soft_empty_cache"), str(mm.calls))
eq("models: ComfyUI's own node-cache flags are set, the same two its Free button sets",
   (q.flags.get("unload_models"), q.flags.get("free_memory")), (True, True))
eq("models: the VRAM actually recovered is measured, not claimed", r["freed_bytes"], 7 * 1024 ** 3)
ok("models: and reported in GB", "7.0GB" in CM.describe(r), CM.describe(r))
ok("models: not marked busy", r["busy"] is False and r["ran"] is True, str(r))

mm, q = FakeMM(), FakeQueue()
install(mm, q)
CM.free_comfy_memory("cache")
ok("cache: the blocks are released", any(c.startswith("soft_empty_cache") for c in mm.calls), str(mm.calls))
ok("cache: but no model is evicted", "unload_all_models" not in mm.calls, str(mm.calls))
eq("cache: and the node cache is left alone", q.flags, {})

# --- the busy case, which is the dangerous one ---------------------------------
mm, q = FakeMM(), FakeQueue(running=[("job", "id")])
install(mm, q)
r = CM.free_comfy_memory("models")
ok("busy: nothing is unloaded out from under a running sampler",
   "unload_all_models" not in mm.calls and not mm.calls, str(mm.calls))
eq("busy: the flags are still set, so ComfyUI frees it between jobs, safely",
   (q.flags.get("unload_models"), q.flags.get("free_memory")), (True, True))
ok("busy: and the caller is told why", r["busy"] is True and "실행 중" in r["detail"], str(r))

mm, q = FakeMM(), FakeQueue(raises=True)
install(mm, q)
CM.free_comfy_memory("models")
ok("an unreadable queue counts as busy — guessing 'idle' is the guess that breaks a render",
   not mm.calls, str(mm.calls))

# --- the node, which IS the running job ----------------------------------------
mm, q = FakeMM(), FakeQueue(running=[("job", "id")])
install(mm, q)
r = CM.free_comfy_memory("models", during_execution=True)
ok("during execution: the busy check does not block the caller that is the job",
   "unload_all_models" in mm.calls, str(mm.calls))
eq("during execution: the node cache of the graph now running is left alone", q.flags, {})
eq("during execution: it is not reported as busy", r["busy"], False)

# --- surviving old and broken ComfyUI builds -------------------------------------
mm, q = FakeMM(old_signature=True), FakeQueue()
install(mm, q)
r = CM.free_comfy_memory("models")
ok("a build whose soft_empty_cache takes no argument still gets its cache emptied",
   "soft_empty_cache(False)" in mm.calls, str(mm.calls))
ok("and the call is reported as having run", r["ran"], str(r))


class Exploding(FakeMM):
    def unload_all_models(self):
        raise RuntimeError("torch is unhappy")


install(Exploding(), FakeQueue())
r = CM.free_comfy_memory("models")
ok("a failed cleanup is reported, and says the generation continues",
   r["ran"] is True and "생성은 계속" in r["detail"], str(r))

install(None, None)

# --- who it runs for --------------------------------------------------------------
def cfg(**kw):
    return R._llm_settings({"llm": kw})


ok("a local LM Studio gets the VRAM", R.wants_comfy_memory(cfg(backend="lmstudio")))
ok("so does Ollama", R.wants_comfy_memory(cfg(backend="ollama")))
ok("turning it off means off", not R.wants_comfy_memory(cfg(backend="lmstudio", free_vram="off")))
ok("Gemini is a cloud API — evicting local checkpoints only costs the next render",
   not R.wants_comfy_memory(cfg(backend="gemini")))
ok("a remote OpenAI-compatible address is somebody else's GPU",
   not R.wants_comfy_memory(cfg(backend="openai_compat", base_url="https://openrouter.ai/api/v1")))
ok("a local address on a nonstandard port still counts as local",
   R.wants_comfy_memory(cfg(backend="openai_compat", base_url="http://127.0.0.1:9999/v1")))
ok("the subscription CLIs are cloud too", not R.wants_comfy_memory(cfg(backend="claude_cli")))
ok("custom_cli may well wrap a local runner, so it gets the VRAM",
   R.wants_comfy_memory(cfg(backend="custom_cli")))
ok("an address that will not parse is not assumed local",
   not R.wants_comfy_memory(cfg(backend="openai_compat", base_url="::::")))

# --- the setting -------------------------------------------------------------------
eq("default: freeing models", cfg()["free_vram"], "models")
eq("an explicit choice is kept", cfg(free_vram="cache")["free_vram"], "cache")
eq("off is kept", cfg(free_vram="off")["free_vram"], "off")
eq("junk falls back to the default, never to something more destructive",
   cfg(free_vram="rm -rf")["free_vram"], "models")
ok("the dialog is told the modes exist", "free_vram_modes" in (_PACK / "server_routes.py").read_text())

if failures:
    print(f"\n✗ {len(failures)} failed, {passed} passed\n")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"✓ all {passed} ComfyUI memory assertions passed")
