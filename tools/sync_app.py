#!/usr/bin/env python3
"""Rebuild everything this pack vendors from the H3 Prompt Maker web app.

Usage:  python3 tools/sync_app.py /path/to/minimax-h3-prompt-maker-google-studio-ai-v3

Two things are copied out of the web app, and they must move together:

  h3_prompts.py   the system prompts, extracted from prompts.ts
  web/app/        the overlay UI, built from the same React source

ComfyUI users clone this repo and expect it to work with no npm, so the built
bundle is committed. That only stays honest if it is regenerated — never edited
in place — which is what this script is for. Run it after any web app change,
then commit whatever it produced.
"""

import hashlib
import json
import shutil
import subprocess
import sys
import pathlib
import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Only what a browser is served. Anything else in dist-comfy is a build leftover.
KEEP_SUFFIXES = {".html", ".js", ".css", ".map", ".svg", ".png", ".ico", ".woff2", ".json"}


def git(src: pathlib.Path, *args: str) -> str:
    """A git field from the web app checkout, or "" when it is not a repo."""
    try:
        out = subprocess.run(["git", "-C", str(src), *args],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 — provenance is best-effort, never fatal
        return ""


def write_provenance(src: pathlib.Path, dist: pathlib.Path) -> None:
    """Stamp SYNC_SOURCE.json with the web app revision these artifacts came from."""
    prompts = src / "prompts.ts"
    bundle = sorted(p.name for p in (dist / "assets").glob("*.js"))
    stamp = {
        "webapp_commit": git(src, "rev-parse", "HEAD"),
        "webapp_subject": git(src, "log", "-1", "--format=%s"),
        # A sync from a checkout with uncommitted edits produces a bundle whose
        # source is not in any repository. That is worth saying out loud.
        "webapp_dirty": bool(git(src, "status", "--porcelain")),
        "prompts_sha256": (hashlib.sha256(prompts.read_bytes()).hexdigest()
                           if prompts.is_file() else ""),
        "bundle_js": bundle,
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
                                 .replace(microsecond=0).isoformat(),
    }
    (ROOT / "SYNC_SOURCE.json").write_text(
        json.dumps(stamp, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    where = stamp["webapp_commit"][:8] or "(git 아님)"
    print(f"   출처 기록: {where}" + (" — 커밋되지 않은 변경 포함!" if stamp["webapp_dirty"] else ""))


def main(webapp: str) -> int:
    src = pathlib.Path(webapp).expanduser().resolve()
    if not (src / "package.json").is_file():
        print(f"not a web app checkout: {src}", file=sys.stderr)
        return 1

    print("== 1/3  프롬프트 추출")
    subprocess.run([sys.executable, str(ROOT / "tools" / "extract_prompts.py"), str(src / "prompts.ts")],
                   check=True)

    print("== 2/3  오버레이 앱 빌드 (npm run build:comfy)")
    subprocess.run(["npm", "run", "build:comfy"], cwd=src, check=True)

    dist = src / "dist-comfy"
    index = dist / "index.html"
    if not index.is_file():
        print(f"build produced no index.html in {dist}", file=sys.stderr)
        return 1

    print("== 3/3  web/app 갱신")
    out = ROOT / "web" / "app"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    copied = 0
    for path in sorted(dist.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in KEEP_SUFFIXES:
            continue
        target = out / path.relative_to(dist)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1

    # Record where these two generated artifacts came from. Nothing in this
    # repository can see the web app, so without a stamp a bundle that is three
    # commits behind looks exactly like a fresh one — which is how the overlay
    # silently shipped stale. check_bundle.py reads this back and prints it.
    write_provenance(src, dist)

    html = index.read_text(encoding="utf-8")
    # The build script already checks this, but the copy is what actually ships.
    for bad in ("cdn.tailwindcss.com", "aistudiocdn.com", 'href="/index.css"'):
        if bad in html:
            print(f"copied bundle still references {bad}", file=sys.stderr)
            return 1

    print(f"\n완료 — {copied}개 파일, {sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) // 1024} KB")
    print("git status 로 확인하고 h3_prompts.py 와 web/app 을 함께 커밋하세요.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
