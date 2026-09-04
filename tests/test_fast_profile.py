"""Run: python3 tests/test_fast_profile.py"""
import importlib.util
import pathlib
import sys

PACK = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "h3fast", PACK / "__init__.py", submodule_search_locations=[str(PACK)])
module = importlib.util.module_from_spec(spec)
sys.modules["h3fast"] = module
spec.loader.exec_module(module)
P = __import__("h3fast.h3_prompts", fromlist=["x"])

full = P.build_system_prompt("ref2va", 10, False, prompt_profile="full")
fast = P.build_system_prompt("ref2va", 10, False, prompt_profile="fast")

checks = {
    "fast profile is materially smaller": len(fast) < len(full) * 0.45,
    "full profile remains available": len(full) > 30_000,
    "mode reaches fast profile": "REF2VA" in fast,
    "duration reaches fast profile": "10" in fast,
    "identity carrier rule survives": "Identity lives in `<Subject N>`" in fast,
    "six-section contract survives": all(x in fast for x in (
        "subject_definitions:", "summary:", "retention_analysis:",
        "detailed_description:", "overall_soundscape:", "non_diegetic_music:")),
    "frame grid survives": "17k+5" in fast and "362" in fast,
    "one-shot rule survives": "Do not ask questions" in fast,
    # Small models copy the Korean request template's bullet headers into the
    # output; the Full profile forbids it and the Fast profile must too.
    "korean template-header rule survives": '"[1] 장면:"' in fast and "Never echo" in fast,
}

failed = [name for name, ok in checks.items() if not ok]
for name, ok in checks.items():
    print(("  ✓ " if ok else "  ✗ ") + name)
print(f"full {len(full):,} chars -> fast {len(fast):,} chars")
sys.exit(1 if failed else 0)
