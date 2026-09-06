#!/usr/bin/env python3
"""Verify the committed overlay bundle is present and self-contained.

web/app is a build artifact checked into git so ComfyUI users need no npm. That
convenience is also the risk: nothing rebuilds it automatically, and a bundle
that reaches for a CDN renders unstyled on an offline machine — which is most
ComfyUI machines. Run from the repo root, or let CI run it.
"""

import json
import pathlib
import re
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
    # Not "some .js exists" — the exact file index.html asks for. A rebuild
    # renames the bundle, so a half-committed sync leaves a script tag pointing
    # at a file that is not there and the overlay loads a blank page.
    referenced = re.findall(r'<script[^>]+src="([^"]+)"', html)
    if not referenced:
        problems.append("index.html 에 번들 script 태그가 없습니다")
    for src in referenced:
        if src.startswith(("http://", "https://", "//")):
            problems.append(f"index.html 이 외부 스크립트를 참조합니다: {src}")
            continue
        if not (APP / src.lstrip("./")).is_file():
            problems.append(f"index.html 이 가리키는 {src} 가 web/app 에 없습니다 "
                            "— tools/sync_app.py 를 다시 실행하세요")

# Provenance. Nothing here can reach the web app, so the only way a reviewer
# can tell a three-commits-behind bundle from a fresh one is this stamp.
stamp_path = ROOT / "SYNC_SOURCE.json"
stamp = {}
if not stamp_path.is_file():
    problems.append("SYNC_SOURCE.json 이 없습니다 — tools/sync_app.py 로 번들을 다시 만드세요")
else:
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        problems.append(f"SYNC_SOURCE.json 을 읽을 수 없습니다: {exc}")
    else:
        built = sorted(p.name for p in (APP / "assets").glob("*.js"))
        if stamp.get("bundle_js") and sorted(stamp["bundle_js"]) != built:
            problems.append(
                f"SYNC_SOURCE.json 이 기록한 번들({stamp['bundle_js']})과 실제 파일({built})이 "
                "다릅니다 — 프롬프트와 번들 중 한쪽만 다시 만든 상태입니다")

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
commit = str(stamp.get("webapp_commit") or "")
if commit:
    subject = str(stamp.get("webapp_subject") or "")
    print(f"  출처: 웹앱 {commit[:8]} {subject[:60]}  ({stamp.get('generated_at', '?')})")
if stamp.get("webapp_dirty"):
    print("  ⚠️  커밋되지 않은 웹앱 변경에서 빌드되었습니다 — 이 번들의 소스는 어디에도 없습니다")
