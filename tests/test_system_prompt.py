"""Does the UI node's request actually carry the H3 system prompt?

Run: python3 tests/test_system_prompt.py
     H3_WEBAPP=/path/to/webapp python3 tests/test_system_prompt.py   (adds parity)


Not "is the string present in the repo" — the question is whether the exact
bytes reach the model. A fake OpenAI-compatible server captures the real
request body that llm_backends sends.
"""
import importlib.util, json, pathlib, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

PACK = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("h3pack", PACK / "__init__.py",
                                              submodule_search_locations=[str(PACK)])
pack = importlib.util.module_from_spec(spec); sys.modules["h3pack"] = pack
spec.loader.exec_module(pack)
R = importlib.import_module("h3pack.server_routes")
P = importlib.import_module("h3pack.h3_prompts")

captured = {}
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        b = json.dumps({"data": [{"id": "fake-model"}]}).encode()
        self.send_response(200); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        captured.update(json.loads(self.rfile.read(n)))
        b = json.dumps({"choices":[{"message":{"content":"```\nok\n```\nlength 243 (10.13 s)"}}]}).encode()
        self.send_response(200); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

srv = HTTPServer(("127.0.0.1", 3399), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()

# Exactly what the overlay posts to /api/generate-prompt, run through the same
# code the route uses.
body = {
    "minimaxStyle": "ref2va", "duration": 10, "isNSFW": False,
    "promptText": "골목을 걷는 인물", "voiceDirection": "낮고 허스키한 목소리",
    "customSystemPrompt": "모든 장면에 비를 추가할 것",
    "llm": {"backend": "openai_compat", "base_url": "http://127.0.0.1:3399/v1", "model": "fake-model"},
}
sysprompt = P.build_system_prompt(
    "ref2va", 10.0, False, camera_instruction="", custom_directives=body["customSystemPrompt"])
user_text = R._build_user_text(body, 0)
cfg = R._llm_settings(body)
from h3pack.llm_backends import call_llm
out = call_llm(cfg["backend"], cfg["base_url"], cfg["model"], "", "", sysprompt, user_text)
srv.shutdown()

msgs = captured.get("messages", [])
sys_msg = next((m["content"] for m in msgs if m.get("role") == "system"), "")
usr_msg = next((m["content"] for m in msgs if m.get("role") == "user"), "")

checks = [
  ("system 역할 메시지가 전송됨", bool(sys_msg)),
  ("H3 프롬프트 엔지니어 지시문", "prompt engineer for **MiniMax H3**" in sys_msg),
  ("Ref2VA 6개 섹션 규격", "retention_analysis" in sys_msg and "non_diegetic_music" in sys_msg),
  ("정체성은 <Subject N>에만 (최다 오류 방지)", "never in a standalone `<Picture N>`" in sys_msg),
  ("Symptom→Fix 표에도 같은 규칙", "never a standalone" in sys_msg),
  ("17k+5 프레임 그리드", "17k+5" in sys_msg),
  ("CFG 1 · 네거티브 프롬프트 없음", "no negative prompt" in sys_msg.lower()),
  ("Symptom → Fix 표", "Symptom" in sys_msg and "Fix" in sys_msg),
  ("좌우(laterality) 락", "laterality" in sys_msg.lower()),
  ("SFW 지시문 (isNSFW=False)", "STRICT SFW MODE" in sys_msg),
  ("NSFW 어휘 미포함", "unrestricted/mature" not in sys_msg),
  ("사용자 커스텀 지침 주입", "모든 장면에 비를 추가할 것" in sys_msg),
  ("목표 길이 10초 반영", "10" in sys_msg),
  ("user 메시지에 장면 요청", "골목을 걷는 인물" in usr_msg),
  ("user 메시지에 목소리 지시", "낮고 허스키한 목소리" in usr_msg),
]
print(f"system 프롬프트 길이: {len(sys_msg):,}자 / user: {len(usr_msg):,}자\n")
bad = 0
for name, ok in checks:
    print(("  ✓ " if ok else "  ✗ ") + name); bad += (not ok)

# Parity with the web app is not a sampled string — the extractor regenerates
# h3_prompts.py from prompts.ts, and the committed file must come back byte for
# byte or the two frontends are sending different instructions.
import os, subprocess
webapp = os.environ.get("H3_WEBAPP", "/home/user/minimax-h3-prompt-maker-google-studio-ai-v3")
prompts_ts = pathlib.Path(webapp) / "prompts.ts"
print()
if prompts_ts.is_file():
    before = (PACK / "h3_prompts.py").read_bytes()
    subprocess.run([sys.executable, str(PACK / "tools" / "extract_prompts.py"), str(prompts_ts)],
                   check=True, capture_output=True)
    after = (PACK / "h3_prompts.py").read_bytes()
    same = before == after
    print(("  ✓ " if same else "  ✗ ") + f"웹앱 prompts.ts에서 재생성 시 바이트 동일 ({len(after):,} bytes)")
    bad += not same
else:
    print(f"  – 동기화 검사 건너뜀 (웹앱 체크아웃 없음: {prompts_ts})")
sys.exit(1 if bad else 0)
