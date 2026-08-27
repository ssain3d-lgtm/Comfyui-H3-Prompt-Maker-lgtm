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
import traceback

from .h3_prompts import build_system_prompt, nearest_grid_frames
from .llm_backends import (
    AUTO_MODEL, BACKEND_NAMES, LLMError, PRESET_BASE_URLS, PRESET_CLI_COMMANDS,
    THINKING_MODES, UNLOAD_MODES, call_llm, clamp_max_tokens, discover_local_models,
    normalize_backend, probe_backend, resolve_backend, warm_up_model,
)

PREFIX = "/h3_prompt_maker"
APP_DIR = pathlib.Path(__file__).parent / "web" / "app"

# The overlay is same-origin with ComfyUI, which has no auth of its own. These
# routes must therefore never read a path the caller supplies, and never run a
# command the caller supplies — cli_command comes from the node's own widget,
# which is the same trust level as the workflow itself.
_ALLOWED_EXT = {".html", ".js", ".css", ".map", ".svg", ".png", ".ico", ".woff2", ".json"}

MAX_BODY_BYTES = 64 * 1024 * 1024  # reference media arrives inline as data URLs


def _safe_asset(rel: str):
    """Resolve rel inside APP_DIR or return None. Rejects traversal and odd types."""
    target = (APP_DIR / rel.lstrip("/")).resolve()
    try:
        target.relative_to(APP_DIR.resolve())
    except ValueError:
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

    roles = body.get("imageRoles")
    if image_count:
        described = []
        for i in range(image_count):
            role = roles[i].strip() if isinstance(roles, list) and i < len(roles) and isinstance(roles[i], str) else ""
            described.append(f"<Picture {i + 1}>" + (f" — {role}" if role else ""))
        lines.append("Reference pictures supplied: " + ", ".join(described))

    if sheet_count:
        n = int(body.get("videoFrameCount") or 8)
        # "<Video 1> ... <Video 1>" reads like two clips. One clip gets one tag.
        tags = "<Video 1>" if sheet_count == 1 else f"<Video 1> ... <Video {sheet_count}>"
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
            f"{audio_count} reference audio clip(s) are attached: <Audio 1> ... <Audio {audio_count}>. "
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
        "thinking": (str(llm.get("thinking") or "auto")
                     if str(llm.get("thinking") or "auto") in THINKING_MODES else "auto"),
        "unload_after": (str(llm.get("unload_after") or "now")
                         if str(llm.get("unload_after") or "now") in UNLOAD_MODES else "now"),
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
            "preset_base_urls": PRESET_BASE_URLS,
            "preset_cli_commands": PRESET_CLI_COMMANDS,
            "models": await _offthread(discover_local_models),
        })

    @routes.post(PREFIX + "/api/probe")
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

    @routes.post(PREFIX + "/api/edit-image")
    async def edit_image(request):
        # Kept so a stale bundle gets a real answer instead of a 404 page.
        return _json({"error": "이미지 편집은 Gemini 전용 기능입니다. "
                               "ComfyUI에서는 인페인트 노드를 사용하세요."}, status=501)

    @routes.post(PREFIX + "/api/generate-prompt")
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

        system_prompt = build_system_prompt(
            submode, seconds, is_nsfw,
            camera_instruction=camera,
            custom_directives=str(body.get("customSystemPrompt") or ""),
            remake=remake,
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
        cfg = _llm_settings(body)
        # A CLI backend takes stdin only, so pictures cannot travel with it.
        send_images = (images + sheets) if not cfg["backend"].endswith("_cli") else []

        # If the last run was told to unload, put the selected model back before
        # the real generation. LM Studio v0.4+ has a native load endpoint; other
        # servers and older LM Studio builds use the one-token fallback.
        warm_note = None
        if cfg["unload_after"] != "keep" and not cfg["backend"].endswith("_cli"):
            try:
                _k, _u, warm_model, _c = resolve_backend(
                    cfg["backend"], cfg["base_url"], cfg["model"], "", cfg["server_model"])
            except Exception:  # noqa: BLE001 — the real call reports it better
                warm_model = ""
            if warm_model:
                warm = await _offthread(warm_up_model, cfg["backend"], cfg["base_url"],
                                        cfg["api_key"], warm_model)
                # A failure here is not fatal: the real call may still succeed,
                # and if it does not, its own error is the more accurate one.
                if not warm.get("ok"):
                    warm_note = warm.get("detail")

        try:
            text = await _offthread(
                call_llm,
                cfg["backend"], cfg["base_url"], cfg["model"], cfg["api_key"], cfg["cli_command"],
                system_prompt, user_text, images_base64=send_images,
                temperature=cfg["temperature"], server_model=cfg["server_model"],
                max_tokens=cfg["max_tokens"], thinking=cfg["thinking"],
                unload_after=cfg["unload_after"],
                audios_base64=audios if not cfg["backend"].endswith("_cli") else None,
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
                      "suggestedFrames": nearest_grid_frames(seconds)})

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
