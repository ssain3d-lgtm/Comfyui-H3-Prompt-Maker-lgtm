"""
ComfyUI nodes: MiniMax H3 Prompt Architect / H3 Prompt Remake.

The system prompts are extracted verbatim from the H3 Prompt Maker web app,
so both frontends generate identical instructions. Output wires directly
into ComfyUI's MiniMax H3 text/conditioning nodes (prompt STRING + frames INT).
"""

import base64
import io
import json
import re

from .h3_prompts import build_system_prompt, nearest_grid_frames
from .llm_backends import call_llm, AUTO_MODEL, BACKEND_NAMES, discover_local_models

SUBMODES = ["ref2va", "t2va", "i2va", "fl2va", "l2va"]
DURATIONS = ["5s (124f)", "6s (158f)", "8s (192f)", "10s (243f)", "12s (294f)", "15s (362f)",
             "20s (2 segments)", "30s (2 segments)", "45s (3 segments)", "60s (4 segments)"]
CONTENT_MODES = ["SFW", "NSFW"]
STRENGTHS = ["subtle", "medium", "reimagine"]
SOURCE_TYPES = ["h3_output", "user_written"]

_LENGTH_RE = re.compile(r"length\s+(\d+)\s*\(\s*(\d+(?:\.\d+)?)\s*s\s*\)", re.IGNORECASE)
# \r?\n so CRLF responses parse; the closing fence is optional so a truncated
# response (max_tokens) still yields the body instead of leaking backticks.
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\r?\n(.*?)(?:```|\Z)", re.DOTALL)
_OPEN_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\r?\n")
# Reasoning models wrap their scratchpad in these; it is not the answer.
_THINK_RE = re.compile(r"<(think|reasoning)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
_H3_SECTIONS = ("subject_definitions", "summary", "retention_analysis", "detailed_description",
                "integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")


def _count_sections(text):
    return sum(1 for name in _H3_SECTIONS
               if re.search(r"^\s*%s\s*:" % name, text, re.MULTILINE | re.IGNORECASE))
_SEGMENT_HEADER_RE = re.compile(r"^[ \t]*Prompt\s+\d+\s*[\u2014\u2013-].*$", re.MULTILINE)
_KOREAN_SEP = "--- KOREAN TRANSLATION ---"

# What <Picture N> means per mode — mirrors the web app. A single generic
# "identity/first shot keyframe" line forced two competing identity carriers in
# Ref2VA and inverted first/last frame in L2VA.
PICTURE_SEMANTICS = {
    "ref2va": "These images are CASTING references, not frames. Cite them INSIDE the relevant "
              "<Subject N> definition as the appearance source and give them NO standalone entry in "
              "subject_definitions or retention_analysis. Identity lives in <Subject N>, never in a "
              "standalone <Picture N>. Only an image with an explicit first/last/key-frame role gets its own entry.",
    "i2va": "<Picture 1> is the exact visual state at 0.00 seconds — frame 0 of [Shot 1]. Further images are "
            "appearance references only: cite them inside <Subject N> and give them no frame entry.",
    "fl2va": "<Picture 1> is the FIRST frame (0.00 s) and <Picture 2> is the LAST frame of the clip. Further "
             "images are appearance references only.",
    "l2va": "<Picture 1> is the FINAL frame the shot lands on — it belongs to the LAST shot, not to [Shot 1]. "
            "Do not describe it as an opening frame.",
    "t2va": "T2VA renders from text alone and has no image reference slot. Do NOT emit <Picture N> tags; write "
            "anything you observe into the shot body as unattributed description.",
}


def _duration_seconds(duration_choice):
    return int(duration_choice.split("s")[0])


def _images_to_base64(images, limit=9):
    """ComfyUI IMAGE tensor [B,H,W,C] float 0..1 -> list of base64 PNG strings."""
    if images is None:
        return []
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return []
    batch = images.cpu().numpy() if hasattr(images, "cpu") else images
    total = min(len(batch), limit)
    out, failures = [], []
    for i in range(total):
        try:
            arr = np.clip(batch[i], 0.0, 1.0)
            if arr.ndim == 3 and arr.shape[-1] == 1:      # grayscale -> RGB
                arr = np.repeat(arr, 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[-1] == 4:    # drop alpha
                arr = arr[..., :3]
            buf = io.BytesIO()
            Image.fromarray((arr * 255.0).astype("uint8")).save(buf, format="PNG")
            out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        except Exception as e:
            failures.append(f"#{i + 1}: {e}")
    if failures:
        # Silently returning fewer images shifts every <Picture N> label after the gap
        raise RuntimeError("H3 Prompt Maker: could not convert " + str(len(failures))
                           + " of " + str(total) + " images — " + "; ".join(failures))
    return out


def _parse_llm_output(text, fallback_seconds):
    """Split an answer into paste-ready prompt(s), a frame count and a Korean summary.

    Handles the shapes that broke the first version: CRLF, unterminated fences,
    <think> scratchpads, a length figure mentioned in prose, and multi-segment
    answers whose Korean summaries used to swallow every later segment.
    """
    # Same recovery as the overlay's parser: a reasoning model that wrote the
    # whole prompt inside <think> and only a note outside has produced the
    # answer, just in the wrong place. Discarding the block would throw it away.
    raw_text = text or ""
    reasoning = "\n\n".join(m.group(2) for m in _THINK_RE.finditer(raw_text))
    without_think = _THINK_RE.sub("", raw_text).strip()
    if _count_sections(without_think) == 0 and _count_sections(reasoning) >= 2:
        # The length line sits outside the block and drives the frame count, so
        # carry it across; the note beside it is not part of the prompt.
        _len = _LENGTH_RE.search(without_think)
        text = (reasoning.strip() + ("\n" + _len.group(0) if _len else "")).strip()
    else:
        text = without_think

    headers = list(_SEGMENT_HEADER_RE.finditer(text))
    if len(headers) >= 2:
        chunks = []
        for i, h in enumerate(headers):
            start = h.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            chunks.append((h.group(0).strip(), text[start:end]))
    else:
        chunks = [(headers[0].group(0).strip() if headers else None, text)]

    segments, korean_parts = [], []
    for header, body in chunks:
        english, _, korean = body.partition(_KOREAN_SEP)
        if korean.strip():
            korean_parts.append(korean.strip())

        # length: prefer the header, then the English body — never the Korean prose
        m = (_LENGTH_RE.search(header) if header else None) or _LENGTH_RE.search(english)

        blocks = [b for b in _FENCE_RE.findall(english) if b.strip()]
        if blocks:
            prompt = "\n\n".join(b.strip() for b in blocks)
        else:
            opener = _OPEN_FENCE_RE.search(english)
            prompt = english[opener.end():] if opener else english
        prompt = _LENGTH_RE.sub("", prompt).replace("```", "").strip()
        if prompt:
            segments.append({"prompt": prompt, "frames": int(m.group(1)) if m else None, "header": header})

    if not segments:
        body = text.partition(_KOREAN_SEP)[0].strip()
        return body, nearest_grid_frames(fallback_seconds), "", body, 0

    # `prompt` is what gets wired into the H3 text encoder, so it carries THIS render
    # only — segment 1, with no header or length noise. The remaining segments stay
    # available through the sequence output.
    first = segments[0]
    sequence = "\n\n".join(
        ((s["header"] + "\n") if s["header"] else f"--- Prompt {i + 1} ---\n") + s["prompt"]
        for i, s in enumerate(segments)
    ) if len(segments) > 1 else first["prompt"]
    frames = first["frames"] or nearest_grid_frames(fallback_seconds)
    return first["prompt"], frames, "\n\n".join(korean_parts), sequence, len(segments)


def _build_user_content(scene_request, dialogue, voice_direction, submode,
                        duration_seconds, image_count, video_ref_note, audio_ref_note,
                        remake_source="", remake_source_type="h3"):
    remake_active = bool(remake_source.strip())
    if remake_active:
        head = (f'Remake direction note from the user:\n"{scene_request}"'
                if scene_request.strip()
                else "Remake the source prompt below with a different feel along the selected axes.")
    else:
        head = (f'User Scene Concept / Request:\n"{scene_request}"'
                if scene_request.strip()
                else "Analyze the attached visual assets and generate a cinematic video prompt.")
    parts = [head]

    if image_count > 0:
        parts.append(
            f"[ATTACHED PICTURE ASSETS]\nTotal Pictures attached: {image_count}.\n"
            f"Label them sequentially <Picture 1> … <Picture {image_count}> in the exact order attached.\n"
            + PICTURE_SEMANTICS.get(submode, PICTURE_SEMANTICS["ref2va"])
        )

    parts.append(
        "[MINIMAX H3 USER INPUTS]\n"
        f"Mode: {submode}\n"
        f"Action/Scene Request: {scene_request}\n"
        f"Dialogue/Spoken Script: {dialogue or 'NONE'}\n"
        f"Speaker Voice Direction: {voice_direction or 'NONE'}\n"
        f"Target Video Duration: {duration_seconds} seconds."
    )

    if submode == "ref2va":
        # Express reference scoping STRUCTURALLY. H3 runs at CFG 1, where a prohibition
        # sentence in the body amplifies exactly what it forbids.
        if video_ref_note.strip():
            parts.append(
                f'Video Reference Asset (<Video 1>): "{video_ref_note}". Express this structurally, never as a '
                "prohibition sentence in the prompt body: merge it positively in subject_definitions "
                '("<Subject 1> is the ... whose motion and camera path come from <Video 1>") and add one scoped '
                'retention_analysis line ("<Video 1> (motion and camera path): weak_reference - only the movement '
                'and pacing are followed; none of its people, wardrobe, location or lighting appear.").'
            )
        if audio_ref_note.strip():
            parts.append(
                f'Audio Reference Asset (<Audio 1>): "{audio_ref_note}". Express this structurally: state in '
                'subject_definitions that "<Audio 1> is the voice-timbre and timing reference for <Subject 1> (S1)" '
                'and give it a documented marker line in retention_analysis ("<Audio 1>: reference - its vocal timbre '
                'and delivery guide the dialogue without copying the original signal."). To keep music out write '
                "`non_diegetic_music: N/A`, never a sentence forbidding music."
            )
    elif image_count > 0 or video_ref_note.strip() or audio_ref_note.strip():
        parts.append(
            f"NOTE: {submode.upper()} runs on the fl2va checkpoint, which has no video or audio reference slot. "
            "Do NOT emit <Video N> or <Audio N> tags — they would point at labels the runtime never injects. "
            "Describe intended motion in the shot body and intended sound in overall_soundscape / non_diegetic_music."
        )

    if remake_active:
        label = ("[REMAKE SOURCE PROMPT — a user-authored prompt/scene description to remake into MiniMax H3 format]"
                 if remake_source_type == "custom"
                 else "[REMAKE SOURCE PROMPT — the previous MiniMax H3 prompt to remake]")
        parts.append(f'{label}\n"""\n{remake_source.strip()}\n"""\n'
                     "Remake it exactly per the REMAKE MODE DIRECTIVE in the system instruction.")

    return "\n\n".join(parts)


_BACKEND_INPUTS = {
    "backend": (BACKEND_NAMES, {
        "default": "lmstudio",
        "tooltip": "lmstudio/ollama/llamacpp/vllm = local OpenAI-compatible servers on their "
                   "standard ports (1234/11434/8080/8000) — nothing else to type. "
                   "gemini = Google Gemini API (needs a GEMINI_API_KEY; model defaults to "
                   "gemini-2.5-flash — the web app's Google models). "
                   "openai_compat = any other OpenAI-compatible address (OpenRouter, "
                   "remote server) via base_url. claude_cli/gemini_cli/codex_cli = "
                   "subscription CLIs. custom_cli = your own stdin->stdout command."}),
    "base_url": ("STRING", {"default": "",
        "tooltip": "Leave empty to use the selected backend's standard address. "
                   "Fill in only for openai_compat or a non-standard port."}),
    "model": ("STRING", {"default": "",
        "tooltip": "Model name typed by hand. The server_model dropdown below wins "
                   "when it is not (auto). Empty = the server's loaded/default model "
                   "(gemini backend: gemini-2.5-flash)."}),
    "api_key": ("STRING", {"default": "", "password": True,
        "tooltip": "Only needed for paid endpoints (Gemini, OpenRouter, OpenAI). "
                   "Local servers ignore it. Leave empty to read the environment instead "
                   "(gemini backend: GEMINI_API_KEY / GOOGLE_API_KEY; others: OPENAI_API_KEY / "
                   "OPENROUTER_API_KEY) — a key typed here is saved into the workflow JSON "
                   "and into the metadata of every image made with it."}),
    "cli_command": ("STRING", {"default": "",
        "tooltip": "CLI backends only. Leave empty to use the preset command "
                   "(claude -p --output-format text / gemini -p / codex exec); "
                   "required for custom_cli."}),
    "temperature": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.05}),
    "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": True,
        "tooltip": "Cache buster: change it to regenerate the same inputs."}),
}

def _server_model_widget():
    """Dropdown of models found on local servers right now. (auto) = use the model field."""
    return ([AUTO_MODEL] + discover_local_models(), {
        "default": AUTO_MODEL,
        "tooltip": "Models discovered on local servers (LM Studio/Ollama/llama.cpp/vLLM). "
                   "Start the server, then refresh the browser to repopulate. "
                   "(auto) = follow the model field above."})


class H3PromptArchitect:
    """Builds a complete MiniMax H3 prompt (T2VA/I2VA/FL2VA/L2VA/Ref2VA) from a scene request."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "scene_request": ("STRING", {"multiline": True, "default": ""}),
            "submode": (SUBMODES, {"default": "ref2va"}),
            "duration": (DURATIONS, {"default": "10s (243f)"}),
            "content_mode": (CONTENT_MODES, {"default": "SFW"}),
        }
        required.update(_BACKEND_INPUTS)
        return {
            "required": required,
            "optional": {
                "images": ("IMAGE",),
                "dialogue": ("STRING", {"multiline": True, "default": ""}),
                "voice_direction": ("STRING", {"default": ""}),
                "camera_direction": ("STRING", {"default": ""}),
                "video_ref_note": ("STRING", {"default": ""}),
                "audio_ref_note": ("STRING", {"default": ""}),
                "custom_directives": ("STRING", {"multiline": True, "default": ""}),
                "prompt_profile": (["fast", "full"], {"default": "fast",
                    "tooltip": "fast = 렌더 핵심 규칙만 보내 입력 처리 단축; full = 전체 H3 가이드."}),
                "send_images_to_llm": ("BOOLEAN", {"default": True}),
                "server_model": _server_model_widget(),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, server_model=None):
        # server_model is a live list — a saved workflow may name a model that is
        # not loaded right now, which must not invalidate the whole graph.
        return True

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING", "INT")
    RETURN_NAMES = ("prompt", "length_frames", "korean_summary", "all_segments", "segment_count")
    FUNCTION = "generate"
    CATEGORY = "H3 Prompt Maker"

    def generate(self, scene_request, submode, duration, content_mode,
                 backend, base_url, model, api_key, cli_command, temperature, seed,
                 images=None, dialogue="", voice_direction="", camera_direction="",
                 video_ref_note="", audio_ref_note="", custom_directives="",
                 prompt_profile="fast", send_images_to_llm=True, server_model=AUTO_MODEL):
        seconds = _duration_seconds(duration)
        is_nsfw = content_mode == "NSFW"
        is_http = not str(backend).endswith("_cli") and backend != "cli"
        all_b64 = _images_to_base64(images) if images is not None else []
        images_b64 = all_b64 if (send_images_to_llm and is_http) else []
        image_count = len(all_b64)

        system_prompt = build_system_prompt(
            submode, seconds, is_nsfw,
            camera_instruction=camera_direction.strip(),
            custom_directives=custom_directives,
            prompt_profile=prompt_profile,
        )
        user_content = _build_user_content(
            scene_request, dialogue, voice_direction, submode, seconds,
            image_count, video_ref_note, audio_ref_note,
        )
        raw = call_llm(backend, base_url, model, api_key, cli_command,
                       system_prompt, user_content, images_base64=images_b64,
                       temperature=temperature if not is_nsfw else max(temperature, 0.9),
                       seed=seed, server_model=server_model)
        prompt, frames, korean, sequence, seg_count = _parse_llm_output(raw, seconds)
        return (prompt, frames, korean, sequence, seg_count)


class H3PromptRemake:
    """Remakes an existing prompt (H3 output or user-written) with a different feel."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "source_prompt": ("STRING", {"multiline": True, "default": ""}),
            "source_type": (SOURCE_TYPES, {"default": "h3_output"}),
            "direction_note": ("STRING", {"multiline": True, "default": ""}),
            "strength": (STRENGTHS, {"default": "medium"}),
            "axis_mood_lighting": ("BOOLEAN", {"default": False}),
            "axis_location": ("BOOLEAN", {"default": False}),
            "axis_wardrobe": ("BOOLEAN", {"default": False}),
            "axis_camera": ("BOOLEAN", {"default": False}),
            "axis_time_season": ("BOOLEAN", {"default": False}),
            "axis_sound_music": ("BOOLEAN", {"default": False}),
            "axis_overall_tone": ("BOOLEAN", {"default": True}),
            "submode": (SUBMODES, {"default": "ref2va"}),
            "duration": (DURATIONS, {"default": "10s (243f)"}),
            "content_mode": (CONTENT_MODES, {"default": "SFW"}),
        }
        required.update(_BACKEND_INPUTS)
        return {
            "required": required,
            "optional": {
                "images": ("IMAGE",),
                "dialogue": ("STRING", {"multiline": True, "default": ""}),
                "voice_direction": ("STRING", {"default": ""}),
                "custom_directives": ("STRING", {"multiline": True, "default": ""}),
                "prompt_profile": (["fast", "full"], {"default": "fast",
                    "tooltip": "fast = 렌더 핵심 규칙만 보내 입력 처리 단축; full = 전체 H3 가이드."}),
                "send_images_to_llm": ("BOOLEAN", {"default": True}),
                "server_model": _server_model_widget(),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, server_model=None):
        return True

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING", "INT")
    RETURN_NAMES = ("prompt", "length_frames", "korean_summary", "all_segments", "segment_count")
    FUNCTION = "remake"
    CATEGORY = "H3 Prompt Maker"

    def remake(self, source_prompt, source_type, direction_note, strength,
               axis_mood_lighting, axis_location, axis_wardrobe, axis_camera,
               axis_time_season, axis_sound_music, axis_overall_tone,
               submode, duration, content_mode,
               backend, base_url, model, api_key, cli_command, temperature, seed,
               images=None, dialogue="", voice_direction="", custom_directives="",
               prompt_profile="fast", send_images_to_llm=True, server_model=AUTO_MODEL):
        if not source_prompt.strip():
            raise ValueError("H3 Prompt Remake: source_prompt is empty — paste the prompt to remake.")

        seconds = _duration_seconds(duration)
        is_nsfw = content_mode == "NSFW"
        axes = [name for flag, name in [
            (axis_mood_lighting, "mood_lighting"),
            (axis_location, "location"),
            (axis_wardrobe, "wardrobe"),
            (axis_camera, "camera"),
            (axis_time_season, "time_season"),
            (axis_sound_music, "sound_music"),
            (axis_overall_tone, "overall_tone"),
        ] if flag]
        src_type = "custom" if source_type == "user_written" else "h3"

        is_http = not str(backend).endswith("_cli") and backend != "cli"
        all_b64 = _images_to_base64(images) if images is not None else []
        images_b64 = all_b64 if (send_images_to_llm and is_http) else []
        image_count = len(all_b64)

        system_prompt = build_system_prompt(
            submode, seconds, is_nsfw,
            custom_directives=custom_directives,
            remake={"axes": axes, "strength": strength, "source_type": src_type},
            prompt_profile=prompt_profile,
        )
        user_content = _build_user_content(
            direction_note, dialogue, voice_direction, submode, seconds,
            image_count, "", "",
            remake_source=source_prompt, remake_source_type=src_type,
        )
        # remakes are creative variation work — keep the higher temperature like the web app
        raw = call_llm(backend, base_url, model, api_key, cli_command,
                       system_prompt, user_content, images_base64=images_b64,
                       temperature=max(temperature, 0.9), seed=seed, server_model=server_model)
        prompt, frames, korean, sequence, seg_count = _parse_llm_output(raw, seconds)
        return (prompt, frames, korean, sequence, seg_count)


class H3PromptMakerUI:
    """The web app's own UI, opened over the ComfyUI canvas.

    The node itself carries no settings — the overlay holds them, and "이 노드에
    적용" writes the finished result into the hidden `result` widget. Execution
    then just emits what was applied, so pressing Queue never silently spends
    another LLM call or changes a prompt the user already approved. web/h3_maker.js
    draws the two buttons and hides these widgets.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # NOT multiline: ComfyUI backs a multiline STRING with a real DOM
        # textarea, and a DOM element is not hidden by the canvas-side tricks
        # the extension uses — one of these three was being rendered on the
        # node as a stray input box. A plain STRING is drawn on the canvas and
        # disappears completely when its height is zeroed. Nobody types in
        # these; the extension writes JSON into them.
        hidden_str = lambda: ("STRING", {"default": ""})
        return {
            "required": {
                # Filled by the overlay; the canvas never shows them.
                "state": hidden_str(),   # the form, so a saved workflow reopens as it was
                "llm": hidden_str(),     # backend/model chosen in the ⚙️ dialog
                "result": hidden_str(),  # the applied output, as JSON
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # These widgets are written by the overlay, not typed by a user; a
        # half-written workflow must not invalidate the whole graph.
        return True

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING", "INT")
    RETURN_NAMES = ("prompt", "length_frames", "korean_summary", "all_segments", "segment_count")
    FUNCTION = "emit"
    CATEGORY = "H3 Prompt Maker"

    def emit(self, state="", llm="", result=""):
        try:
            data = json.loads(result) if str(result).strip() else {}
        except (ValueError, TypeError):
            data = {}
        prompt = str(data.get("prompt") or "").strip()
        if not prompt:
            raise RuntimeError(
                "H3 Prompt Maker: 아직 적용된 프롬프트가 없습니다. "
                "노드의 '🎬 프롬프트 메이커 열기'를 눌러 생성한 뒤 '이 노드에 적용'을 누르세요."
            )
        frames = data.get("lengthFrames")
        if not isinstance(frames, int) or frames <= 0:
            frames = nearest_grid_frames(10)
        return (
            prompt,
            frames,
            str(data.get("koreanSummary") or ""),
            str(data.get("allSegments") or prompt),
            int(data.get("segmentCount") or 1),
        )


NODE_CLASS_MAPPINGS = {
    "H3PromptArchitect": H3PromptArchitect,
    "H3PromptRemake": H3PromptRemake,
    "H3PromptMakerUI": H3PromptMakerUI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3PromptArchitect": "MiniMax H3 Prompt Architect 🎬",
    "H3PromptRemake": "MiniMax H3 Prompt Remake 🔄",
    "H3PromptMakerUI": "MiniMax H3 Prompt Maker (UI) 🖥️",
}
