"""Regression tests for the one function in this pack that is pure and fragile.

Every case here is an output shape that produced a wrong prompt or a wrong frame
count at some point. `_parse_llm_output`'s INT output drives the H3 sampler, so a
bad frame number renders the whole clip at the wrong speed.

Run: python3 -m pytest tests/ -q      (or: python3 tests/test_parse.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "h3_nodes_undertest",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nodes.py"),
)


def _load():
    """Import nodes.py without its package-relative imports."""
    import types
    pkg = types.ModuleType("h3pkg")
    pkg.__path__ = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
    sys.modules["h3pkg"] = pkg
    spec = importlib.util.spec_from_file_location(
        "h3pkg.nodes", os.path.join(pkg.__path__[0], "nodes.py"), submodule_search_locations=pkg.__path__)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h3pkg.nodes"] = mod
    spec.loader.exec_module(mod)
    return mod


nodes = _load()
_raw_parse = nodes._parse_llm_output


def parse(text, secs):
    """(prompt, frames, korean) — the first three of the node's five outputs."""
    return _raw_parse(text, secs)[:3]


def parse_full(text, secs):
    return _raw_parse(text, secs)


def _clean(prompt):
    assert "```" not in prompt, f"code fence leaked: {prompt[:80]!r}"
    assert "length " not in prompt.lower() or "(" not in prompt, f"length line leaked: {prompt[:80]!r}"


def test_fenced_with_length():
    p, f, k = parse("Here you go:\n```\nsubject_definitions:\nA\n```\nlength 243 (10.13 s)", 10)
    _clean(p); assert f == 243 and p.startswith("subject_definitions:") and k == ""


def test_no_fence():
    p, f, _ = parse("subject_definitions:\nA\n\nlength 124 (5.17 s)", 5)
    _clean(p); assert f == 124


def test_crlf():
    p, f, _ = parse("ok:\r\n```\r\nsubject_definitions:\r\nA\r\n```\r\nlength 192 (8.00 s)", 8)
    _clean(p); assert f == 192


def test_unterminated_fence_is_not_leaked():
    # A response cut off by max_tokens used to return the opening backticks
    p, f, _ = parse("```\nsubject_definitions:\nA truncated", 10)
    _clean(p); assert p.startswith("subject_definitions:") and f == 243


def test_reasoning_block_is_not_the_answer():
    raw = ("<think>\n```\nthis reasoning block is far longer than the real prompt body\n```\n</think>\n"
           "```\nsubject_definitions:\nREAL\n```\nlength 90 (3.75 s)")
    p, f, _ = parse(raw, 4)
    _clean(p); assert "REAL" in p and "reasoning" not in p and f == 90


def test_length_in_korean_prose_is_ignored():
    raw = "```\nA\n```\nlength 243 (10.13 s)\n--- KOREAN TRANSLATION ---\n길이는 length 5 (0.21 s) 입니다."
    p, f, k = parse(raw, 10)
    assert f == 243, "a length figure inside the Korean summary must not win"
    assert "길이는" in k


def test_korean_outside_the_fence_is_kept():
    p, f, k = parse("```\nA\n```\nlength 124 (5.17 s)\n--- KOREAN TRANSLATION ---\n요약", 5)
    assert k == "요약"


def test_multi_segment_prompt_is_this_render_only():
    raw = ("Prompt 1 — <Picture 1> — length 362 (15.08 s)\n```\nSEG ONE\n```\n"
           "Prompt 2 — <Picture 1> — length 158 (6.58 s)\n```\nSEG TWO\n```")
    p, f, _, sequence, count = parse_full(raw, 20)
    _clean(p)
    # `prompt` feeds the sampler, so it must be segment 1 alone and match length_frames
    assert p.strip() == "SEG ONE"
    assert f == 362, "the frame count must be this render's, not a blanket 362"
    # nothing is lost: the full plan is on its own output
    assert count == 2 and "SEG ONE" in sequence and "SEG TWO" in sequence


def test_multi_segment_with_a_korean_summary_per_segment():
    raw = ("Prompt 1 — length 362 (15.08 s)\n```\nS1\n```\n--- KOREAN TRANSLATION ---\n요약1\n"
           "Prompt 2 — length 362 (15.08 s)\n```\nS2\n```\n--- KOREAN TRANSLATION ---\n요약2\n"
           "Prompt 3 — length 243 (10.13 s)\n```\nS3\n```\n"
           "Prompt 4 — length 124 (5.17 s)\n```\nS4\n```")
    p, f, k, sequence, count = parse_full(raw, 60)
    assert count == 4
    for seg in ("S1", "S2", "S3", "S4"):
        assert seg in sequence, f"{seg} was dropped"
    assert p.strip() == "S1" and f == 362
    assert "요약1" in k and "요약2" in k


def test_empty_response_falls_back_to_the_grid():
    p, f, _ = parse("", 12)
    assert f == 294 and p == ""


def test_grid_rounds_up():
    # 6 s x 24 = 144 -> 158, matching the guide's round-up rule
    assert nodes.nearest_grid_frames(6) == 158
    assert nodes.nearest_grid_frames(10) == 243
    assert nodes.nearest_grid_frames(999) == 362


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"  FAIL  {name}: {e}")
    print("\nALL PASS" if not failures else f"\n{failures} FAILED")
    sys.exit(1 if failures else 0)
