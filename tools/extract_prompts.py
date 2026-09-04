#!/usr/bin/env python3
"""Regenerate h3_prompts.py from the H3 Prompt Maker web app's prompts.ts.

Usage:  python3 tools/extract_prompts.py /path/to/webapp/prompts.ts

Keeps this node pack's system prompts byte-identical to the web app
(minimax-h3-prompt-maker-google-studio-ai-v3). Run it whenever the web
app's H3 prompt changes, then commit the regenerated h3_prompts.py.

The templates used to live in server.ts alongside the express routes; the
web app moved every prompt string into prompts.ts so this tool has exactly
one file to read. A prompt added to a route handler instead of prompts.ts
is invisible here and silently desyncs the node pack — the assertions
below catch a missing section, not a missing file.
"""

import re
import sys
import pathlib

TOKEN_MAP = {
    'selectedMode.toUpperCase()': '«MODE»',
    'duration': '«DURATION»',
    'cameraInstructionText': '«CAMERA»',
    'safetyInstruction': '«SAFETY»',
    'remakeDirective': '«REMAKE»',
    'multiSegmentDirective': '«MULTISEG»',
    'segmentCount': '«SEGMENTS»',
    'axesText': '«AXES»',
    'strengthText': '«STRENGTH»',
    'sourceHandling': '«SOURCE_HANDLING»',
}


def main(prompts_ts_path, out_override=None):
    src = open(prompts_ts_path, encoding="utf-8").read()

    def tokenize_and_unescape(t):
        def sub(m):
            expr = m.group(1).strip()
            assert expr in TOKEN_MAP, f"unknown interpolation: {expr}"
            return TOKEN_MAP[expr]
        t = re.sub(r'(?<!\\)\$\{([^}]+)\}', sub, t)
        return (t.replace('\\\\', '\x00').replace('\\`', '`')
                 .replace('\\${', '${').replace('\x00', '\\'))

    def find_unescaped_backtick(s, start):
        i = start
        while True:
            i = s.index('`', i)
            if s[i - 1] != '\\':
                return i
            i += 1

    def extract_template(anchor):
        a = src.index(anchor)
        open_bt = src.rindex('`', 0, a)
        close_bt = find_unescaped_backtick(src, a)
        return tokenize_and_unescape(src[open_bt + 1:close_bt])

    base = extract_template('You are a prompt engineer for **MiniMax H3**')
    fast = extract_template('[H3 FAST PROFILE — ONE-SHOT OUTPUT]')
    multiseg = extract_template('*MULTI-SEGMENT DIRECTIVE (ACTIVE')
    remake = extract_template('*REMAKE MODE DIRECTIVE (ACTIVE):*')
    sh_custom = extract_template('- The source is a USER-AUTHORED')
    sh_h3 = extract_template('- PRESERVE VERBATIM (copy the exact wording')
    preamble_nsfw = extract_template('IMPORTANT DIRECTIVE FOR CREATIVE ARTISTIC PROMPTING:')
    preamble_sfw = extract_template('CONTEXT: This request is a technical prompt-engineering task')

    m = re.search(r'const safetyInstruction = isNSFW\s*\?\s*"([^"]+)"\s*:\s*"([^"]+)";', src)
    safety_nsfw, safety_sfw = m.group(1), m.group(2)

    def extract_dict(name):
        block = re.search(name + r'[^=]*=\s*\{(.*?)\n\};', src, re.S).group(1)
        return dict(re.findall(r"(\w+): '([^']*)'", block))

    axes = extract_dict('REMAKE_AXIS_DESCRIPTIONS')
    strengths = extract_dict('REMAKE_STRENGTH_DESCRIPTIONS')
    assert len(axes) == 7 and len(strengths) == 3

    for marker in ['«MODE»', '«DURATION»', '«CAMERA»', '«SAFETY»', '«REMAKE»', '«MULTISEG»',
                   'Writing a motion strip into the description', '## Limits',
                   'Voice control', 'REQUIREMENT COVERAGE', '17k+5',
                   'App execution context']:
        assert marker in base, f"missing in base: {marker}"
    for marker in ['«MODE»', '«DURATION»', '«CAMERA»', '«SAFETY»', '«REMAKE»', '«MULTISEG»',
                   'subject_definitions:', 'integrated_multimodal_description:',
                   'overall_soundscape:', 'non_diegetic_music:', '17k+5']:
        assert marker in fast, f"missing in fast profile: {marker}"
    assert '«SEGMENTS»' in multiseg

    out_path = pathlib.Path(__file__).resolve().parent.parent / 'h3_prompts.py'
    py = '''"""
MiniMax H3 system prompts, extracted verbatim from the H3 Prompt Maker web app
(minimax-h3-prompt-maker-google-studio-ai-v3, prompts.ts) so both stay in sync.
Regenerate with tools/extract_prompts.py — do not edit the constants by hand.
Prompt guide adapted from https://github.com/teskor-hub/minimax-h3-skill (MIT, (c) 2026 teskor).
"""

SAFETY_NSFW = {safety_nsfw!r}

SAFETY_SFW = {safety_sfw!r}

PREAMBLE_NSFW = {preamble_nsfw!r}

PREAMBLE_SFW = {preamble_sfw!r}

BASE_TEMPLATE = {base!r}

FAST_BASE_TEMPLATE = {fast!r}

MULTISEG_TEMPLATE = {multiseg!r}

REMAKE_TEMPLATE = {remake!r}

SOURCE_HANDLING_H3 = {sh_h3!r}

SOURCE_HANDLING_CUSTOM = {sh_custom!r}

REMAKE_AXIS_DESCRIPTIONS = {axes!r}

REMAKE_STRENGTH_DESCRIPTIONS = {strengths!r}

GRID_17K5 = [5, 22, 39, 56, 73, 90, 107, 124, 141, 158, 175, 192, 209, 226, 243, 260, 277, 294, 311, 328, 345, 362]

CUSTOM_DIRECTIVES_BLOCK = (
    "\\n=========================================\\n"
    "[USER CUSTOM SYSTEM DIRECTIVES / \\uc0ac\\uc6a9\\uc790 \\ucee4\\uc2a4\\ud140 \\uc2dc\\uc2a4\\ud15c \\uc9c0\\uce68]\\n"
    "{{directives}}\\n"
    "*Strictly follow and enforce the above user custom instructions throughout all prompt sections and outputs.*\\n"
    "=========================================\\n"
)


def build_remake_directive(submode, axes, strength, source_type):
    axes = [a for a in (axes or []) if a]
    axes_text = "; ".join(REMAKE_AXIS_DESCRIPTIONS.get(a, a) for a in (axes or ["overall_tone"]))
    strength_text = REMAKE_STRENGTH_DESCRIPTIONS.get(strength, REMAKE_STRENGTH_DESCRIPTIONS["medium"])
    handling = SOURCE_HANDLING_CUSTOM if source_type == "custom" else SOURCE_HANDLING_H3
    return (REMAKE_TEMPLATE
            .replace("\\u00abSOURCE_HANDLING\\u00bb", handling)
            .replace("\\u00abAXES\\u00bb", axes_text)
            .replace("\\u00abSTRENGTH\\u00bb", strength_text)
            .replace("\\u00abMODE\\u00bb", submode.upper()))


def build_system_prompt(submode, duration_seconds, is_nsfw, camera_instruction="",
                        custom_directives="", remake=None, prompt_profile="full"):
    """Mirror of the web app's getSystemInstruction + safeGuardedSystemInstruction for minimax_h3."""
    camera_text = "Camera Setting Request: " + camera_instruction + "\\n" if camera_instruction else ""
    remake_directive = ""
    if remake:
        remake_directive = build_remake_directive(
            submode, remake.get("axes"), remake.get("strength", "medium"), remake.get("source_type", "h3"))
    import math
    is_multi = duration_seconds > 15.08
    segments = math.ceil(duration_seconds / 15.08)
    multiseg_directive = ""
    if is_multi:
        multiseg_directive = (MULTISEG_TEMPLATE
                              .replace("\\u00abSEGMENTS\\u00bb", str(segments))
                              .replace("\\u00abDURATION\\u00bb", str(duration_seconds)))
    # These must agree with the segment count, or the model gets contradictory
    # instructions about how many code blocks to emit.
    closing = (
        ("Output exactly " + str(segments) + " consecutive prompts for target mode **"
         + submode.upper() + "**, each in its own code block, each preceded by its "
         "`Prompt k \\u2014 <references needed> \\u2014 length X (Y.YY s)` header. "
         "Do not merge them into one block and do not stop after the first.")
        if is_multi else
        ("Output the single complete prompt for target mode **" + submode.upper()
         + "**, followed by `length X (Y.YY s)`.")
    )
    block_rule = "one code block per segment (never all segments in one block)" if is_multi else "a single code block"
    template = FAST_BASE_TEMPLATE if prompt_profile == "fast" else BASE_TEMPLATE
    base = (template
            .replace("\\u00abMODE\\u00bb", submode.upper())
            .replace("\\u00abDURATION\\u00bb", str(duration_seconds))
            .replace("\\u00abCAMERA\\u00bb", camera_text)
            .replace("\\u00abSAFETY\\u00bb", SAFETY_NSFW if is_nsfw else SAFETY_SFW)
            .replace("\\u00abMULTISEG\\u00bb", multiseg_directive)
            .replace("\\u00abCLOSING\\u00bb", closing)
            .replace("\\u00abBLOCKRULE\\u00bb", block_rule)
            .replace("\\u00abREMAKE\\u00bb", remake_directive))
    preamble = PREAMBLE_NSFW if is_nsfw else PREAMBLE_SFW
    custom_block = CUSTOM_DIRECTIVES_BLOCK.format(directives=custom_directives.strip()) if custom_directives.strip() else ""
    return preamble + "\\n" + custom_block + "\\n" + base


def nearest_grid_frames(duration_seconds):
    """Snap UP to the next 17k+5 value, matching the guide (144 -> 158, not 141)."""
    target = duration_seconds * 24
    return next((f for f in GRID_17K5 if f >= target), GRID_17K5[-1])
'''.format(
        safety_nsfw=safety_nsfw, safety_sfw=safety_sfw,
        preamble_nsfw=preamble_nsfw, preamble_sfw=preamble_sfw,
        base=base, fast=fast, multiseg=multiseg, remake=remake, sh_h3=sh_h3, sh_custom=sh_custom,
        axes=axes, strengths=strengths,
    )
    target = pathlib.Path(out_override) if out_override else out_path
    target.write_text(py, encoding="utf-8")
    print(f"{target.name} regenerated ({len(py)} chars) from {prompts_ts_path}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    # A second argument writes elsewhere, so a parity check does not have to
    # overwrite the committed file to find out whether it matches.
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
