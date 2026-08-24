#!/usr/bin/env python3
"""Rebuild everything this pack vendors from the H3 Prompt Maker web app.

Usage:  python3 tools/sync_app.py /path/to/minimax-h3-prompt-maker-google-studio-ai-v2

Two things are copied out of the web app, and they must move together:

  h3_prompts.py   the system prompts, extracted from prompts.ts
  web/app/        the overlay UI, built from the same React source

ComfyUI users clone this repo and expect it to work with no npm, so the built
bundle is committed. That only stays honest if it is regenerated — never edited
in place — which is what this script is for. Run it after any web app change,
then commit whatever it produced.
"""

import shutil
import subprocess
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Only what a browser is served. Anything else in dist-comfy is a build leftover.
KEEP_SUFFIXES = {".html", ".js", ".css", ".map", ".svg", ".png", ".ico", ".woff2", ".json"}


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
