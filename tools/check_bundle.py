#!/usr/bin/env python3
"""Verify the committed overlay bundle is present and self-contained.

web/app is a build artifact checked into git so ComfyUI users need no npm. That
convenience is also the risk: nothing rebuilds it automatically, and a bundle
that reaches for a CDN renders unstyled on an offline machine — which is most
ComfyUI machines. Run from the repo root, or let CI run it.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = ROOT / "web" / "app"

problems = []

index = APP / "index.html"
if not index.is_file():
    problems.append("web/app/index.html 이 없습니다 — tools/sync_app.py 를 실행하세요")
else:
    html = index.read_text(encoding="utf-8")
    for bad, why in (
        ("cdn.tailwindcss.com", "Tailwind CDN — 오프라인에서 스타일이 통째로 빠집니다"),
        ("aistudiocdn.com", "AI Studio import map — 오프라인에서 앱이 뜨지 않습니다"),
        ('href="/index.css"', "절대 경로 — ComfyUI 하위 경로에서 404"),
        ('src="/assets', "절대 경로 — ComfyUI 하위 경로에서 404"),
    ):
        if bad in html:
            problems.append(f"index.html 에 {bad} 가 남아 있습니다 ({why})")
    if "./tailwind.css" not in html:
        problems.append("빌드된 tailwind.css 가 링크되어 있지 않습니다")
    if not (APP / "tailwind.css").is_file():
        problems.append("web/app/tailwind.css 가 없습니다")
    else:
        css = (APP / "tailwind.css").read_text(encoding="utf-8")
        if len(css) < 10_000:
            problems.append(f"tailwind.css 가 너무 작습니다 ({len(css)}B) — 스캔이 실패했을 수 있습니다")
    if not list((APP / "assets").glob("*.js")) if (APP / "assets").is_dir() else True:
        problems.append("web/app/assets 에 번들 JS가 없습니다")

ext = ROOT / "web" / "h3_maker.js"
if not ext.is_file():
    problems.append("web/h3_maker.js 가 없습니다 — 노드에 버튼이 그려지지 않습니다")

if problems:
    print("\n✗ 오버레이 번들 점검 실패\n")
    for p in problems:
        print("  - " + p)
    sys.exit(1)
size = sum(f.stat().st_size for f in APP.rglob("*") if f.is_file()) // 1024
print(f"✓ 오버레이 번들 정상 — {size} KB, 외부 참조 없음")
