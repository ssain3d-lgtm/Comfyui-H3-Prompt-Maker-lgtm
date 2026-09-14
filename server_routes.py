"""HTTP routes that back the in-ComfyUI prompt maker overlay.

The overlay is the web app's own bundle (built from the same source, see
tools/sync_app.py). It expects the endpoints the Express server gave it, so
they are recreated here on ComfyUI's aiohttp app and answered by llm_backends
instead of Gemini.

Registered on import from __init__.py. If ComfyUI's server module is not
importable — running the tests, or importing the pack standalone — this is a
no-op rather than an error.
"""

import asyncio
import functools
import json
import mimetypes
import pathlib
import time
import urllib.parse
import traceback

from .comfy_memory import DEFAULT_FREE_MODE, FREE_MODES, describe, free_comfy_memory
from .h3_prompts import build_system_prompt, nearest_grid_frames
from .llm_backends import (
    AUTO_MODEL, BACKEND_NAMES, LLMCancelled, LLMError, PRESET_BASE_URLS,
    PRESET_CLI_COMMANDS, StreamCancel, THINKING_MODES, UNLOAD_MODES, call_llm,
    clamp_max_tokens, discover_local_models, is_local_target, normalize_backend,
    probe_backend, resolve_backend, stream_llm, unload_model, warm_up_model,
)

PREFIX = "/h3_prompt_maker"
APP_DIR = pathlib.Path(__file__).parent / "web" / "app"

# The overlay is same-origin with ComfyUI, which has no auth of its own. These
# routes must therefore never read a path the caller supplies, and never run a
# command the caller supplies — cli_command comes from the node's own widget,
# which is the same trust level as the workflow itself.
_ALLOWED_EXT = {".html", ".js", ".css", ".map", ".svg", ".png", ".ico", ".woff2", ".json"}

MAX_BODY_BYTES = 64 * 1024 * 1024  # reference media arrives inline as data URLs
PROMPT_PROFILES = ("fast", "full")


def _safe_asset(rel: str):
    """Resolve rel inside APP_DIR or return None. Rejects traversal and odd types."""
    try:
        target = (APP_DIR / rel.lstrip("/")).resolve()
        target.relative_to(APP_DIR.resolve())
    except ValueError:
        # Traversal, or a path the OS will not even parse: a %00 in the URL
        # made resolve() raise "embedded null byte", which reached the browser
        # as a 500 and a traceback in the ComfyUI console instead of a 404.
        return None
    except OSError:
        return None
    if not target.is_file() or target.suffix.lower() not in _ALLOWED_EXT:
        return None
    return target


def _strip_data_url(value):
    if not isinstance(value, str):
        return None
    if not value.startswith("data:"):
        return value
    # A prefix with no comma is not a data URL at all; partition keeps it whole
    # instead of raising, so one malformed attachment cannot 500 the request.
    head, sep, payload = value.partition(",")
    return payload if sep else value


def _collect(body, *keys):
    out = []
    for key in keys:
        v = body.get(key)
        if isinstance(v, str) and v:
            out.append(v)
        elif isinstance(v, list):
            out.extend([x for x in v if isinstance(x, str) and x])
    return out


def _build_user_text(body, image_count, sheet_count=0, audio_count=0):
    """Same assembly the widget node does, from the overlay's request shape."""
    submode = str(body.get("minimaxStyle") or "ref2va")
    lines = [f"[MINIMAX H3 {submode.upper()} REQUEST]"]
    scene = str(body.get("promptText") or "").strip()
    if scene:
        lines.append(f"Scene / action request:\n{scene}")

    dialogue = str(body.get("ltxNarration") or "").strip()
    if dialogue:
        lines.append(f"Spoken dialogue (wrap each line in <d>...</d>):\n{dialogue}")
    voice = str(body.get("voiceDirection") or "").strip()
    if voice:
        lines.append(f"Speaker voice: {voice}. Express it inside the speaker's <Subject N> "
                     f"definition, never as a standalone instruction.")

    def _labelled(kind, count, key):
        """`<Kind 1> — role, <Kind 2>` for however many actually travelled.

        The overlay collects a per-asset role note for pictures, clips AND audio
        (InputSection), and App.tsx goes to real trouble to keep the video list
        aligned with the clips that survive the size filter. Only imageRoles was
        ever read here, so in ComfyUI every clip and audio note the user typed
        was dropped before the model saw it — while the same note worked in the
        standalone web app.
        """
        supplied = body.get(key)
        out = []
        for i in range(count):
            role = (supplied[i].strip()
                    if isinstance(supplied, list) and i < len(supplied)
                    and isinstance(supplied[i], str) else "")
            out.append(f"<{kind} {i + 1}>" + (f" — {role}" if role else ""))
        return ", ".join(out)

    if image_count:
        lines.append("Reference pictures supplied: " + _labelled("Picture", image_count, "imageRoles"))

    if sheet_count:
        n = int(body.get("videoFrameCount") or 8)
        # "<Video 1> ... <Video 1>" reads like two clips. One clip gets one tag.
        tags = _labelled("Video", sheet_count, "videoRoles")
        lines.append(
            f"Attached after the reference pictures are {sheet_count} contact sheet(s), one per "
            f"reference clip: {tags}, in that order. Each sheet tiles "
            f"{n} frames sampled evenly across that clip, left to right and top to bottom, with the "
            f"frame number and its timestamp printed under each cell. Read them as one moving shot, "
            f"not as separate stills: the change between cells is the camera path, the subject's "
            f"motion arc and the pacing. Describe what moves and how, never the grid itself — the "
            f"words 'contact sheet', 'grid', 'tile' and 'frame number' must not appear in the prompt."
        )

    if audio_count:
        lines.append(
            f"{audio_count} reference audio clip(s) are attached: "
            f"{_labelled('Audio', audio_count, 'audioRoles')}. "
            f"If you can hear them, let what you hear drive overall_soundscape and non_diegetic_music, "
            f"and give each one a documented marker (fully_copy | partially_copy | reference | "
            f"weak_reference) in retention_analysis. If you cannot hear audio at all, say nothing "
            f"about these clips and write the soundscape from the scene and the note below instead — "
            f"never invent what a clip you did not hear contains."
        )

    for label, key in (("Video", "videoRefNote"), ("Audio", "audioRefNote")):
        note = str(body.get(key) or "").strip()
        if note:
            lines.append(f"{label} reference note: {note}. Express this in the structural "
                         f"slots (subject_definitions / retention_analysis), not as a prohibition.")

    if body.get("isRemake"):
        source = str(body.get("remakeSourcePrompt") or "").strip()
        if source:
            lines.append(f"[REMAKE SOURCE PROMPT]\n{source}")
    return "\n\n".join(lines)


def _llm_settings(body):
    llm = body.get("llm") if isinstance(body.get("llm"), dict) else {}
    backend = normalize_backend(str(llm.get("backend") or "lmstudio"))
    if backend not in BACKEND_NAMES:
        backend = "lmstudio"
    try:
        temperature = float(llm.get("temperature", 0.7))
    except (TypeError, ValueError):
        temperature = 0.7
    return {
        "backend": backend,
        "base_url": str(llm.get("base_url") or ""),
        "model": str(llm.get("model") or ""),
        "api_key": str(llm.get("api_key") or ""),
        "cli_command": str(llm.get("cli_command") or ""),
        "temperature": max(0.0, min(2.0, temperature)),
        "server_model": str(llm.get("server_model") or AUTO_MODEL),
        "max_tokens": clamp_max_tokens(llm.get("max_tokens")),
        "thinking": (str(llm.get("thinking") or "off")
                     if str(llm.get("thinking") or "off") in THINKING_MODES else "off"),
        "unload_after": (str(llm.get("unload_after") or "close")
                         if str(llm.get("unload_after") or "close") in UNLOAD_MODES else "close"),
        "prompt_profile": (str(llm.get("prompt_profile") or "fast")
                           if str(llm.get("prompt_profile") or "fast") in PROMPT_PROFILES else "fast"),
        "free_vram": (str(llm.get("free_vram") or DEFAULT_FREE_MODE)
                      if str(llm.get("free_vram") or DEFAULT_FREE_MODE) in FREE_MODES
                      else DEFAULT_FREE_MODE),
    }


async def _read_body(request, limit=MAX_BODY_BYTES):
    """Read the entire request payload, bounded. None when it exceeds `limit`.

    Not request.content.read(limit): that is a StreamReader read, which returns
    whatever happens to be buffered — up to n, and routinely far less. A body
    carrying a base64 image arrives across many chunks, so that call handed back
    the first fragment and json.loads reported an unterminated string at around
    column 292. Every generation with an attachment failed that way.
    """
    chunks, total = [], 0
    async for chunk in request.content.iter_chunked(1 << 16):
        total += len(chunk)
        if total > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


async def _offthread(fn, *args, **kwargs):
    """Run a blocking call without stopping ComfyUI.

    Everything this module reaches for — probe, warm-up, the generation itself,
    model discovery — is synchronous urllib. Called directly from an `async def`
    handler it holds the one event loop ComfyUI runs on for the whole duration:
    measured at 3.01s against a deliberately slow stub, and a local 14B model
    with a 60k token budget holds it for minutes. While it is held the websocket
    drops ("Reconnecting…"), the queue stops, /view and /prompt hang and the
    progress bar freezes — which reads as ComfyUI crashing, not as a slow model.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


def _selected_model(cfg):
    try:
        _kind, _url, model, _command = resolve_backend(
            cfg["backend"], cfg["base_url"], cfg["model"], "", cfg["server_model"])
        return model
    except Exception:  # noqa: BLE001 — the real generation reports the useful error
        return ""


#: Servers that load the model themselves when a request names it, so a
#: pre-flight ping only costs a request — on llama.cpp it can also evict the
#: prompt cache that made the previous prefill cheap. (A llama-server router
#: can still be unloaded after the call; a single-model one and vLLM cannot.)
_ALWAYS_RESIDENT_BACKENDS = {"llamacpp", "vllm"}


#: CLI runners that reach a cloud service. `custom_cli` is not on the list:
#: it is whatever command the machine owner exported, which may well be a
#: local llama.cpp wrapper that wants the VRAM.
_CLOUD_CLI_BACKENDS = {"claude_cli", "gemini_cli", "codex_cli"}


def wants_comfy_memory(cfg):
    """True when this generation runs on THIS GPU, so freeing it helps.

    A cloud endpoint gains nothing from evicting the local checkpoints — it
    only costs the next render the time to load them again.
    """
    if cfg["free_vram"] == "off":
        return False
    backend = normalize_backend(cfg["backend"])
    if backend in _CLOUD_CLI_BACKENDS:
        return False
    if backend.endswith("_cli"):
        return True
    try:
        _kind, url, _model, _cmd = resolve_backend(backend, cfg["base_url"], "", "")
    except Exception:  # noqa: BLE001 — the real call reports the useful error
        return False
    return is_local_target(url)


async def _free_comfy_for_generation(cfg):
    """Hand ComfyUI's VRAM to the LLM before asking it for anything.

    Off the event loop: unload_all_models walks every loaded model and a
    gc.collect over a freshly dropped checkpoint is not instant, and holding
    ComfyUI's one loop there is what makes the websocket say "Reconnecting…".
    """
    if not wants_comfy_memory(cfg):
        return {"ran": False, "busy": False, "detail": "", "freed_bytes": None}
    result = await _offthread(free_comfy_memory, cfg["free_vram"])
    line = describe(result)
    if line:
        print(f"[h3_prompt_maker] {line}", flush=True)
    return result


async def _warm_for_generation(cfg):
    """Ensure modes that can unload have a resident model and time the check/load."""
    if cfg["unload_after"] == "keep" or cfg["backend"].endswith("_cli"):
        return None, {"load_ms": 0.0, "load_detail": "model kept resident"}
    if normalize_backend(cfg["backend"]) in _ALWAYS_RESIDENT_BACKENDS:
        return None, {"load_ms": 0.0, "load_detail": "server process owns the model"}
    model = _selected_model(cfg)
    if not model:
        return None, {"load_ms": 0.0, "load_detail": "automatic model selection"}
    started = time.perf_counter()
    warm = await _offthread(warm_up_model, cfg["backend"], cfg["base_url"],
                            cfg["api_key"], model)
    load_ms = round((time.perf_counter() - started) * 1000, 1)
    return (None if warm.get("ok") else warm.get("detail")), {
        "load_ms": load_ms,
        "load_detail": str(warm.get("detail") or ""),
    }


def prepare_generation(body):
    """Turn one overlay-shaped request body into everything a generation needs.

    Shared by the HTTP route the overlay calls and by the UI-Instant node,
    which rebuilds the same body from the form the overlay saved into the
    node — so both paths ask the model for exactly the same thing.

    Returns a dict: cfg, system_prompt, user_text, send_images, audios, seconds.
    """
    submode = str(body.get("minimaxStyle") or "ref2va")
    try:
        seconds = float(body.get("duration") or 10)
    except (TypeError, ValueError):
        seconds = 10.0
    seconds = max(1.0, min(60.0, seconds))
    is_nsfw = bool(body.get("isNSFW"))

    camera = " ".join(x for x in (str(body.get("cameraPosition") or "").strip(),
                                  str(body.get("cameraAngle") or "").strip()) if x)
    remake = None
    if body.get("isRemake"):
        axes = body.get("remakeAxes")
        remake = {
            "axes": [a for a in axes if isinstance(a, str)] if isinstance(axes, list) else [],
            "strength": str(body.get("remakeStrength") or "medium"),
            "source_type": "custom" if body.get("remakeSourceType") == "custom" else "h3",
        }

    cfg = _llm_settings(body)
    system_prompt = build_system_prompt(
        submode, seconds, is_nsfw,
        camera_instruction=camera,
        custom_directives=str(body.get("customSystemPrompt") or ""),
        remake=remake,
        prompt_profile=cfg["prompt_profile"],
    )

    images = [_strip_data_url(x) for x in _collect(body, "imageBase64", "imagesBase64")]
    images = [x for x in images if x][:9]
    # The OpenAI-compatible chat schema every local backend speaks has no
    # video part, so a clip arrives as one contact sheet of its frames and
    # rides along as an ordinary image. Sheets go after the pictures and
    # keep their own cap, so a third clip can never push out <Picture 9>.
    sheets = [_strip_data_url(x) for x in _collect(body, "videoFramesBase64")]
    sheets = [x for x in sheets if x][:3]
    # Audio only lands anywhere on an omni model (Qwen2-Audio, Qwen2.5/3-Omni).
    # Sending it regardless is right: the call sheds it on rejection, so a
    # text model behaves exactly as before while an omni model gains the clip.
    audios = [_strip_data_url(x) for x in _collect(body, "audioBase64", "audiosBase64")]
    audios = [x for x in audios if x][:3]
    user_text = _build_user_text(body, len(images), len(sheets), len(audios))
    is_cli = cfg["backend"].endswith("_cli")
    # A CLI backend takes stdin only, so pictures cannot travel with it.
    return {
        "cfg": cfg,
        "system_prompt": system_prompt,
        "user_text": user_text,
        "send_images": (images + sheets) if not is_cli else [],
        "audios": audios if not is_cli else None,
        "seconds": seconds,
    }


def _accepts_stream(request):
    headers = getattr(request, "headers", {})
    return "application/x-ndjson" in str(headers.get("Accept", "")).lower()


async def _stream_generation(request, cfg, system_prompt, user_text, send_images,
                             audios, seconds):
    """Bridge a blocking OpenAI SSE stream to browser-friendly NDJSON."""
    from aiohttp import web

    response = web.StreamResponse(status=200, headers={
        "Content-Type": "application/x-ndjson; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        "X-Content-Type-Options": "nosniff",
    })
    await response.prepare(request)
    route_started = time.perf_counter()

    async def write_event(event):
        data = (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        await response.write(data)

    control = StreamCancel()
    future = None
    try:
        freed = {"ran": False}
        if wants_comfy_memory(cfg):
            await write_event({"type": "status", "stage": "freeing",
                               "message": "ComfyUI VRAM·캐시 비우는 중…"})
            freed = await _free_comfy_for_generation(cfg)
        await write_event({"type": "status", "stage": "loading",
                           "message": "모델 준비 중…" if not freed.get("ran")
                                      else f"{describe(freed)} · 모델 준비 중…"})
        warm_note, load_metrics = await _warm_for_generation(cfg)
        if freed.get("ran"):
            load_metrics = {**load_metrics,
                            "comfy_freed_bytes": freed.get("freed_bytes"),
                            "comfy_free_detail": freed.get("detail", "")}
        await write_event({
            "type": "status", "stage": "generating", "message": "프롬프트 생성 중…",
            "metrics": load_metrics,
        })

        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()

        def put(event):
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def run():
            try:
                text, metrics = stream_llm(
                    cfg["backend"], cfg["base_url"], cfg["model"], cfg["api_key"],
                    cfg["cli_command"], system_prompt, user_text,
                    images_base64=send_images, temperature=cfg["temperature"],
                    server_model=cfg["server_model"], max_tokens=cfg["max_tokens"],
                    thinking=cfg["thinking"], unload_after=cfg["unload_after"],
                    audios_base64=audios, on_delta=lambda chunk: put({"type": "delta", "text": chunk}),
                    cancel=control,
                )
                put({"type": "complete", "text": text, "metrics": metrics})
            except LLMCancelled:
                put({"type": "cancelled"})
            except LLMError as exc:
                detail = str(exc)
                if warm_note:
                    detail += ("\n\n모델 준비도 실패했습니다: " + str(warm_note)
                               + "\n⚙️ 모델 연결에서 '모델 로드'를 한 번 눌러보세요.")
                put({"type": "error", "error": detail, "reason": "transient"})
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                put({"type": "error", "error": f"{type(exc).__name__}: {exc}", "reason": "unknown"})

        future = loop.run_in_executor(None, run)
        last_progress = 0.0
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                elapsed = (time.perf_counter() - route_started) * 1000
                if elapsed - last_progress >= 1000:
                    await write_event({"type": "progress", "elapsed_ms": round(elapsed, 1)})
                    last_progress = elapsed
                continue

            kind = event.get("type")
            if kind == "complete":
                text = str(event.pop("text", ""))
                if not text.strip():
                    await write_event({"type": "error", "error": "모델이 빈 응답을 반환했습니다.",
                                       "reason": "transient"})
                else:
                    metrics = dict(event.get("metrics") or {})
                    metrics.update(load_metrics)
                    metrics["generation_ms"] = metrics.get("total_ms")
                    metrics["total_ms"] = round((time.perf_counter() - route_started) * 1000, 1)
                    await write_event({
                        "type": "final", "result": text, "fallback": False,
                        "suggestedFrames": nearest_grid_frames(seconds), "metrics": metrics,
                    })
                break
            if kind == "cancelled":
                await write_event({"type": "cancelled"})
                break
            if kind == "error":
                await write_event(event)
                break
            await write_event(event)
    except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError, RuntimeError):
        # Browser AbortController or closing the overlay lands here. Closing the
        # upstream urllib response wakes its blocking readline immediately.
        control.cancel()
    finally:
        if control.is_set() and future is not None:
            future.cancel()
        try:
            await response.write_eof()
        except Exception:
            pass
    return response


def _cross_site(request):
    """True when a browser sent this request from a different site.

    ComfyUI has no authentication of its own, and these routes have real side
    effects: they load and unload the user's model, and resolve_api_key hands
    an environment key to Google's pinned host with no interaction. A page the
    user happens to have open could POST here — no preflight is needed for a
    text/plain body — and while it could not read the reply, the side effects
    would land, on the user's billable key.

    Only a browser attaches Origin and Sec-Fetch-Site automatically, so an
    absent Origin means a tool (curl, a script, the node itself) and is not a
    CSRF vector. Reject a present-and-different one.
    """
    if str(request.headers.get("Sec-Fetch-Site") or "").lower() == "cross-site":
        return True
    origin = str(request.headers.get("Origin") or "").strip()
    if not origin:
        return False
    # A sandboxed frame or a downloaded file:// page sends "null". The overlay
    # is served by ComfyUI and is same-origin with it, so this never comes from
    # anything legitimate here.
    if origin.lower() == "null":
        return True
    host = str(request.headers.get("Host") or "").strip()
    if not host:
        return False
    try:
        return urllib.parse.urlsplit(origin).netloc.lower() != host.lower()
    except ValueError:
        return True


def _same_site_only(handler):
    """Guard a mutating route against a cross-site browser POST."""
    @functools.wraps(handler)
    async def guarded(request):
        if _cross_site(request):
            return _json({"error": "다른 사이트에서 보낸 요청은 처리하지 않습니다."}, status=403)
        return await handler(request)
    return guarded


def register(routes):
    @routes.get(PREFIX + "/api/health")
    async def health(request):
        return _json({"ok": True, "app_built": (APP_DIR / "index.html").is_file()})

    @routes.get(PREFIX + "/api/backends")
    async def backends(request):
        return _json({
            "backends": BACKEND_NAMES,
            "thinking_modes": THINKING_MODES,
            "unload_modes": UNLOAD_MODES,
            "prompt_profiles": PROMPT_PROFILES,
            "free_vram_modes": FREE_MODES,
            "preset_base_urls": PRESET_BASE_URLS,
            "preset_cli_commands": PRESET_CLI_COMMANDS,
            "models": await _offthread(discover_local_models),
        })

    @routes.post(PREFIX + "/api/probe")
    @_same_site_only
    async def probe(request):
        """Reachability check for the settings dialog's 연결 확인 button."""
        try:
            body = json.loads(await request.text() or "{}")
        except ValueError:
            body = {}
        cfg = _llm_settings({"llm": body})
        return _json(await _offthread(probe_backend, cfg["backend"], cfg["base_url"],
                                      cfg["api_key"], cfg["cli_command"]))

    @routes.post(PREFIX + "/api/load-model")
    @_same_site_only
    async def load_model(request):
        """Ask the server to page the chosen model into memory before it is needed."""
        try:
            body = json.loads(await request.text() or "{}")
        except ValueError:
            body = {}
        cfg = _llm_settings({"llm": body})
        model = str(body.get("model") or "").strip()
        if cfg["backend"].endswith("_cli"):
            return _json({"ok": True, "detail": "CLI 백엔드는 미리 로드할 모델이 없습니다."})
        return _json(await _offthread(warm_up_model, cfg["backend"], cfg["base_url"],
                                      cfg["api_key"], model))

    @routes.post(PREFIX + "/api/unload-model")
    @_same_site_only
    async def unload_selected_model(request):
        """Release the chosen local model on demand or when the overlay closes."""
        try:
            body = json.loads(await request.text() or "{}")
        except ValueError:
            body = {}
        cfg = _llm_settings({"llm": body})
        if cfg["backend"].endswith("_cli"):
            return _json({"ok": True, "supported": False,
                          "detail": "CLI 백엔드는 실행이 끝나면 프로세스도 종료됩니다."})
        model = _selected_model(cfg)
        return _json(await _offthread(unload_model, cfg["backend"], cfg["base_url"],
                                      cfg["api_key"], model))

    @routes.post(PREFIX + "/api/edit-image")
    @_same_site_only
    async def edit_image(request):
        # Kept so a stale bundle gets a real answer instead of a 404 page.
        return _json({"error": "이미지 편집은 Gemini 전용 기능입니다. "
                               "ComfyUI에서는 인페인트 노드를 사용하세요."}, status=501)

    @routes.post(PREFIX + "/api/generate-prompt")
    @_same_site_only
    async def generate_prompt(request):
        try:
            raw = await _read_body(request)
            if raw is None:
                return _json({"error": "요청이 너무 큽니다 (64MB 초과)."}, status=413)
            body = json.loads(raw.decode("utf-8", errors="replace"))
            if not isinstance(body, dict):
                raise ValueError("body must be an object")
        except Exception as exc:
            return _json({"error": f"잘못된 요청입니다: {exc}"}, status=400)

        prep = prepare_generation(body)
        cfg, seconds = prep["cfg"], prep["seconds"]
        system_prompt, user_text = prep["system_prompt"], prep["user_text"]
        send_images, audios = prep["send_images"], prep["audios"]

        if _accepts_stream(request):
            return await _stream_generation(
                request, cfg, system_prompt, user_text, send_images, audios, seconds)

        # JSON compatibility path for old overlay bundles and direct callers.
        freed = await _free_comfy_for_generation(cfg)
        warm_note, load_metrics = await _warm_for_generation(cfg)
        if freed.get("ran"):
            load_metrics = {**load_metrics,
                            "comfy_freed_bytes": freed.get("freed_bytes"),
                            "comfy_free_detail": freed.get("detail", "")}

        try:
            text = await _offthread(
                call_llm,
                cfg["backend"], cfg["base_url"], cfg["model"], cfg["api_key"], cfg["cli_command"],
                system_prompt, user_text, images_base64=send_images,
                temperature=cfg["temperature"], server_model=cfg["server_model"],
                max_tokens=cfg["max_tokens"], thinking=cfg["thinking"],
                unload_after=cfg["unload_after"],
                audios_base64=audios,
            )
        except LLMError as exc:
            detail = str(exc)
            if warm_note:
                detail = (f"{detail}\n\n모델을 먼저 올려보려 했지만 실패했습니다: {warm_note}\n"
                          f"'생성 후'가 언로드로 설정돼 있어 모델이 메모리에서 내려가 있을 수 있습니다. "
                          f"LM Studio라면 JIT 모델 로딩이 켜져 있는지 확인하거나, "
                          f"⚙️ 모델 연결에서 '모델 로드'를 한 번 누르세요.")
            return _json({"error": detail, "reason": "transient"}, status=502)
        except Exception as exc:  # noqa: BLE001 — the overlay must show something actionable
            traceback.print_exc()
            return _json({"error": f"{type(exc).__name__}: {exc}", "reason": "unknown"}, status=502)

        if not str(text).strip():
            return _json({"error": "모델이 빈 응답을 반환했습니다. 다른 모델을 쓰거나 "
                                   "컨텍스트 길이를 늘려보세요.", "reason": "transient"}, status=502)
        return _json({"result": text, "fallback": False,
                      "suggestedFrames": nearest_grid_frames(seconds),
                      "metrics": load_metrics})

    @routes.get(PREFIX + "/app")
    async def app_index(request):
        # The bundle links its assets relatively (./assets/...), so without the
        # trailing slash the browser resolves them one directory too high and
        # the overlay comes up blank. Redirect rather than serve here.
        from aiohttp import web
        raise web.HTTPMovedPermanently(PREFIX + "/app/")

    @routes.get(PREFIX + r"/app/{tail:.*}")
    async def app_asset(request):
        rel = request.match_info.get("tail") or "index.html"
        return await _send(_safe_asset(rel))


# --- aiohttp helpers, imported lazily so the module is testable without ComfyUI

def _json(payload, status=200):
    from aiohttp import web
    return web.json_response(payload, status=status)


async def _send(path):
    from aiohttp import web
    if path is None:
        return web.Response(
            status=404,
            content_type="text/plain",
            text="오버레이 앱이 빌드되어 있지 않습니다. "
                 "tools/sync_app.py 로 web/app 을 생성하세요.",
        )
    ctype, _ = mimetypes.guess_type(path.name)
    return web.FileResponse(path, headers={"Content-Type": ctype or "application/octet-stream"})


def install():
    """Attach the routes to ComfyUI's server. Safe to call when it is absent."""
    try:
        from server import PromptServer
    except Exception:
        return False
    instance = getattr(PromptServer, "instance", None)
    if instance is None or not hasattr(instance, "routes"):
        return False
    register(instance.routes)
    return True
