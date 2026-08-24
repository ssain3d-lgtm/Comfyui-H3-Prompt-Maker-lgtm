"""
ComfyUI nodes: MiniMax H3 Prompt Architect / H3 Prompt Remake.

The system prompts are extracted verbatim from the H3 Prompt Maker web app,
so both frontends generate identical instructions. Output wires directly
into ComfyUI's MiniMax H3 text/conditioning nodes (prompt STRING + frames INT).
"""

import base64
import io
import re

from .h3_prompts import build_system_prompt, nearest_grid_frames
from .llm_backends import call_llm

SUBMODES = ["ref2va", "t2va", "i2va", "fl2va", "l2va"]
DURATIONS = ["5s (124f)", "6s (141f)", "8s (192f)", "10s (243f)", "12s (294f)", "15s (362f)"]
CONTENT_MODES = ["SFW", "NSFW"]
BACKENDS = ["openai_compatible", "cli"]
STRENGTHS = ["subtle", "medium", "reimagine"]
SOURCE_TYPES = ["h3_output", "user_written"]

_LENGTH_RE = re.compile(r"length\s+(\d+)\s*\(\s*(\d+(?:\.\d+)?)\s*s\s*\)", re.IGNORECASE)
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\n(.*?)```", re.DOTALL)


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
    out = []
    try:
        batch = images.cpu().numpy() if hasattr(images, "cpu") else images
        for i in range(min(len(batch), limit)):
            arr = (np.clip(batch[i], 0.0, 1.0) * 255.0).astype("uint8")
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    except Exception as e:
        print(f"[H3 Prompt Maker] image conversion skipped: {e}")
    return out


def _parse_llm_output(text, fallback_seconds):
    """Unwrap the prompt code block, pull out the recommended length line."""
    frames = None
    m = None
    for m in _LENGTH_RE.finditer(text):
        pass
    if m:
        frames = int(m.group(1))

    blocks = _FENCE_RE.findall(text)
    if blocks:
        prompt = max(blocks, key=len).strip()
    else:
        prompt = text.strip()

    # strip a trailing length line the model may have left inside the prompt body
    prompt = _LENGTH_RE.sub("", prompt).rstrip().rstrip("`").rstrip()

    korean = ""
    if "--- KOREAN TRANSLATION ---" in prompt:
        prompt, korean = prompt.split("--- KOREAN TRANSLATION ---", 1)
        prompt, korean = prompt.strip(), korean.strip()

    if frames is None:
        frames = nearest_grid_frames(fallback_seconds)
    return prompt, frames, korean


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
            f"[ATTACHED MULTI-PICTURE ASSETS]\nTotal Pictures attached: {image_count}.\n"
            f"Label them sequentially: <Picture 1>, <Picture 2>, ... <Picture {image_count}> "
            "in the exact order attached. Establish <Picture 1> as the primary subject identity/first shot keyframe."
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
        if video_ref_note.strip():
            parts.append(
                f'Video Reference Asset (<Video 1>): "{video_ref_note}". Explicitly state in the MiniMax prompt: '
                "Use <Video 1> only for body motion, dance timing, facial performance, and camera movement. "
                "Do not copy identity, clothing, background, or visual appearance from <Video 1>."
            )
        if audio_ref_note.strip():
            parts.append(
                f'Audio Reference Asset (<Audio 1>): "{audio_ref_note}". Explicitly state in the MiniMax prompt: '
                "Use <Audio 1> as the exact voice, dialogue timing, emotion, and lip-sync reference. "
                "Preserve original speech rhythm and vocal tone. Do not generate additional dialogue or background music."
            )

    if remake_active:
        label = ("[REMAKE SOURCE PROMPT — a user-authored prompt/scene description to remake into MiniMax H3 format]"
                 if remake_source_type == "custom"
                 else "[REMAKE SOURCE PROMPT — the previous MiniMax H3 prompt to remake]")
        parts.append(f'{label}\n"""\n{remake_source.strip()}\n"""\n'
                     "Remake it exactly per the REMAKE MODE DIRECTIVE in the system instruction.")

    return "\n\n".join(parts)


_BACKEND_INPUTS = {
    "backend": (BACKENDS, {"default": "openai_compatible"}),
    "base_url": ("STRING", {"default": "http://localhost:1234/v1"}),
    "model": ("STRING", {"default": "local-model"}),
    "api_key": ("STRING", {"default": ""}),
    "cli_command": ("STRING", {"default": "claude -p --output-format text"}),
    "temperature": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.05}),
    "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": True}),
}


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
                "send_images_to_llm": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("prompt", "length_frames", "korean_summary")
    FUNCTION = "generate"
    CATEGORY = "H3 Prompt Maker"

    def generate(self, scene_request, submode, duration, content_mode,
                 backend, base_url, model, api_key, cli_command, temperature, seed,
                 images=None, dialogue="", voice_direction="", camera_direction="",
                 video_ref_note="", audio_ref_note="", custom_directives="",
                 send_images_to_llm=True):
        seconds = _duration_seconds(duration)
        is_nsfw = content_mode == "NSFW"
        images_b64 = _images_to_base64(images) if (send_images_to_llm and backend == "openai_compatible") else []
        image_count = len(_images_to_base64(images)) if images is not None else 0

        system_prompt = build_system_prompt(
            submode, seconds, is_nsfw,
            camera_instruction=camera_direction.strip(),
            custom_directives=custom_directives,
        )
        user_content = _build_user_content(
            scene_request, dialogue, voice_direction, submode, seconds,
            image_count, video_ref_note, audio_ref_note,
        )
        raw = call_llm(backend, base_url, model, api_key, cli_command,
                       system_prompt, user_content, images_base64=images_b64,
                       temperature=temperature if not is_nsfw else max(temperature, 0.9),
                       seed=seed)
        prompt, frames, korean = _parse_llm_output(raw, seconds)
        return (prompt, frames, korean)


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
                "send_images_to_llm": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("prompt", "length_frames", "korean_summary")
    FUNCTION = "remake"
    CATEGORY = "H3 Prompt Maker"

    def remake(self, source_prompt, source_type, direction_note, strength,
               axis_mood_lighting, axis_location, axis_wardrobe, axis_camera,
               axis_time_season, axis_sound_music, axis_overall_tone,
               submode, duration, content_mode,
               backend, base_url, model, api_key, cli_command, temperature, seed,
               images=None, dialogue="", voice_direction="", custom_directives="",
               send_images_to_llm=True):
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

        images_b64 = _images_to_base64(images) if (send_images_to_llm and backend == "openai_compatible") else []
        image_count = len(_images_to_base64(images)) if images is not None else 0

        system_prompt = build_system_prompt(
            submode, seconds, is_nsfw,
            custom_directives=custom_directives,
            remake={"axes": axes, "strength": strength, "source_type": src_type},
        )
        user_content = _build_user_content(
            direction_note, dialogue, voice_direction, submode, seconds,
            image_count, "", "",
            remake_source=source_prompt, remake_source_type=src_type,
        )
        # remakes are creative variation work — keep the higher temperature like the web app
        raw = call_llm(backend, base_url, model, api_key, cli_command,
                       system_prompt, user_content, images_base64=images_b64,
                       temperature=max(temperature, 0.9), seed=seed)
        prompt, frames, korean = _parse_llm_output(raw, seconds)
        return (prompt, frames, korean)


NODE_CLASS_MAPPINGS = {
    "H3PromptArchitect": H3PromptArchitect,
    "H3PromptRemake": H3PromptRemake,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3PromptArchitect": "MiniMax H3 Prompt Architect 🎬",
    "H3PromptRemake": "MiniMax H3 Prompt Remake 🔄",
}
