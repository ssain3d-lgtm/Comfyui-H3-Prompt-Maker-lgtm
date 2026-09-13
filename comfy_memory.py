"""Freeing ComfyUI's own VRAM and node cache before a local LLM generation.

One card, two tenants. ComfyUI keeps the checkpoints it last sampled with
resident, and a local LLM asks for several more gigabytes on top — so the
first generation after a render either spills into system RAM and crawls, or
the server answers "failed to allocate". Clicking ComfyUI's own "Free model
and node cache" first is the manual fix; this does it as part of pressing
generate.

Everything here is best-effort and never raises: outside ComfyUI (the test
suite, a standalone import) `comfy.model_management` simply is not importable,
and the pack has to keep working.
"""

import gc

#: off        nothing is freed — the previous behaviour
#: cache      release the torch allocator's cached blocks, keep models resident
#: models     unload every model ComfyUI holds, then release the blocks (default)
FREE_MODES = ["off", "cache", "models"]
DEFAULT_FREE_MODE = "models"


def _model_management():
    try:
        import comfy.model_management as mm  # noqa: PLC0415 — absent outside ComfyUI
    except Exception:  # noqa: BLE001
        return None
    return mm


def _prompt_queue():
    try:
        from server import PromptServer  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    return getattr(getattr(PromptServer, "instance", None), "prompt_queue", None)


def _queue_busy(queue):
    """True when ComfyUI is executing right now, or when that cannot be told.

    Unloading a model out from under a running sampler is the one way this can
    damage someone's render, so a busy queue takes the deferred path instead:
    the flags below are what ComfyUI's own Free button sets, and its worker
    applies them between jobs, where it is safe. An unreadable queue counts as
    busy — guessing "idle" is the guess that breaks a render.
    """
    try:
        running, _pending = queue.get_current_queue()
    except Exception:  # noqa: BLE001
        return True
    return bool(running)


def _free_bytes(mm):
    try:
        return int(mm.get_free_memory(mm.get_torch_device()))
    except Exception:  # noqa: BLE001 — CPU-only installs have no such device
        return None


def _empty_cache(mm):
    try:
        mm.soft_empty_cache(True)
    except TypeError:
        # Older builds take no argument.
        mm.soft_empty_cache()


def free_comfy_memory(mode=DEFAULT_FREE_MODE, during_execution=False):
    """Release what ComfyUI is holding. Returns {ran, busy, detail, freed_bytes}.

    `freed_bytes` is the change in free VRAM across the call, so it is a
    measurement rather than a claim — 0 or None simply means nothing was
    resident, not that the call failed.

    `during_execution` is for a caller that IS the running job — the UI-Instant
    node, executing inside the graph. Two things change for it: the busy check
    would otherwise refuse (it is looking at that very node), and the node
    cache is left alone, because wiping the cache of the graph currently
    running only forces the next queue to redo work it already has.
    """
    result = {"ran": False, "busy": False, "detail": "", "freed_bytes": None}
    if mode not in FREE_MODES or mode == "off":
        result["detail"] = "설정에서 꺼져 있습니다."
        return result

    mm = _model_management()
    if mm is None:
        result["detail"] = "ComfyUI 밖에서 실행 중이라 비울 VRAM이 없습니다."
        return result

    before = _free_bytes(mm)
    notes = []
    queue = _prompt_queue()

    # Ask ComfyUI itself, exactly as its own "Free model and node cache" button
    # does. This is the only way to reach the node output cache: it lives on the
    # PromptExecutor inside the worker thread and nothing outside can touch it.
    if queue is not None and mode == "models" and not during_execution:
        try:
            queue.set_flag("unload_models", True)
            queue.set_flag("free_memory", True)
            notes.append("노드 캐시 비우기 요청")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"노드 캐시 요청 실패: {exc}")

    if queue is not None and not during_execution and _queue_busy(queue):
        result["ran"] = True
        result["busy"] = True
        notes.append("실행 중인 작업이 있어 즉시 해제는 건너뜁니다 (ComfyUI가 작업이 끝나면 처리)")
        result["detail"] = "; ".join(notes)
        return result

    try:
        if mode == "models":
            mm.unload_all_models()
            notes.append("모델 언로드")
        # gc first: unload_all_models drops ComfyUI's references, but the
        # tensors are only returned to the allocator once Python collects them,
        # and emptying the cache before that frees nothing.
        gc.collect()
        _empty_cache(mm)
        notes.append("VRAM 캐시 해제")
    except Exception as exc:  # noqa: BLE001 — a failed cleanup must not fail the generation
        result["ran"] = True
        result["detail"] = f"VRAM 해제 실패(생성은 계속): {exc}"
        return result

    after = _free_bytes(mm)
    if before is not None and after is not None:
        result["freed_bytes"] = max(0, after - before)
    result["ran"] = True
    result["detail"] = "; ".join(notes)
    return result


def describe(result):
    """One human line for the overlay's status area and the ComfyUI console."""
    if not result.get("ran"):
        return ""
    freed = result.get("freed_bytes")
    if freed:
        return f"ComfyUI VRAM {freed / (1024 ** 3):.1f}GB 확보 — {result['detail']}"
    return f"ComfyUI 메모리 정리 — {result['detail']}"
