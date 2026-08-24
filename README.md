# ComfyUI H3 Prompt Maker

MiniMax H3 (비디오+오디오 옴니모달 모델) 전용 **프롬프트 생성 커스텀 노드**입니다.
장면 요청(한국어 가능)과 참조 이미지를 넣으면 LLM이 H3 공식 포맷
(T2VA / I2VA / FL2VA / L2VA / Ref2VA)의 영어 프롬프트를 설계하고,
결과를 **같은 워크플로우의 H3 노드에 바로 와이어로 연결**할 수 있습니다.

[H3 Prompt Maker 웹앱](https://github.com/ssain3d-lgtm/minimax-h3-prompt-maker-google-studio-ai-v2)과
동일한 시스템 프롬프트를 사용합니다 (server.ts에서 그대로 추출).

## 설치

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ssain3d-lgtm/Comfyui-H3-Prompt-Maker-lgtm-.git
# 추가 의존성 없음 (표준 라이브러리 + ComfyUI 기본 패키지만 사용)
```

ComfyUI 재시작 후 노드 검색에서 `H3 Prompt Maker` 카테고리를 찾으세요.

## 노드

### 🎬 MiniMax H3 Prompt Architect
장면 요청 → 완성된 H3 프롬프트.

| 출력 | 용도 |
|---|---|
| `prompt` (STRING) | H3 텍스트 인코딩 노드에 연결 |
| `length_frames` (INT) | H3 length/frames 입력에 연결 (17k+5 그리드, LLM 추천값 파싱) |
| `korean_summary` (STRING) | 한국어 요약 (모델이 제공한 경우) |

주요 입력: 장면 요청 · 세부모드(ref2va/t2va/i2va/fl2va/l2va) · 길이 프리셋 ·
SFW/NSFW · 참조 이미지(IMAGE, 최대 9장 — `<Picture N>` 라벨 자동 부여) ·
대사 · **화자 목소리 묘사** · 카메라 지정 · 비디오/오디오 참조 메모 · 커스텀 지침.

### 🔄 MiniMax H3 Prompt Remake
기존 프롬프트를 "유사하지만 다른 느낌"으로 변주.

- **소스 종류**: `h3_output`(이전 H3 결과물 — 구조·정체성 문구 잠금) /
  `user_written`(직접 작성·외부 프롬프트 — 분석 후 H3 형식으로 재구성)
- **변경 축** 토글 7종: 분위기/조명 · 장소/배경 · 의상 · 카메라 문법 · 시간대/계절 · 사운드/음악 · 전체 톤
- **강도**: subtle / medium / reimagine

## LLM 백엔드

`backend` 드롭다운에서 고르면 표준 주소/명령이 자동으로 적용됩니다
(ComfyUI-LLM-Hub와 같은 방식). 대부분의 경우 **백엔드만 고르고 `server_model`
드롭다운에서 모델을 선택**하면 끝입니다.

### 로컬 HTTP 서버 (프리셋 — base_url 입력 불필요)

| backend | 자동 주소 | 비고 |
|---|---|---|
| `lmstudio` | `http://127.0.0.1:1234/v1` | 기본값. 서버 켜고 브라우저 새로고침하면 `server_model`에 모델 목록이 뜸 |
| `ollama` | `http://127.0.0.1:11434/v1` | |
| `llamacpp` | `http://127.0.0.1:8080/v1` | `llama-server` 기준. `llama-cli` 단발 실행은 매번 모델을 리로드하므로 비추천 |
| `vllm` | `http://127.0.0.1:8000/v1` | |

- **`server_model` 드롭다운**: 위 표준 포트에 떠 있는 서버들의 모델 목록을 자동 조회합니다
  (내 컴퓨터 loopback만 조회 — 원격/유료 주소는 건드리지 않음). `(auto)` = `model` 칸의 값 사용.
  서버를 켠 뒤 브라우저를 새로고침하면 목록이 갱신됩니다.
- 참조 이미지는 **비전 모델**(Qwen-VL, Gemma 3 비전 등)을 로드했을 때만 LLM이 직접 봅니다.
  비전 미지원 모델이면 자동으로 텍스트 전용으로 재시도합니다 (`send_images_to_llm` 끄기 가능).
- 권장 로컬 모델: **Qwen3 14B 이상 인스트럭트** — 형식 준수가 빡빡해서 8B 이하는 자주 깨집니다.

### `openai_compat` — 그 외 OpenAI 호환 주소 (base_url 직접 입력)

| 대상 | base_url | 비고 |
|---|---|---|
| OpenRouter | `https://openrouter.ai/api/v1` | api_key 필요 |
| Gemini (OpenAI 호환) | `https://generativelanguage.googleapis.com/v1beta/openai` | api_key 필요 — 웹앱과 동일 품질 |
| OpenAI / KoboldCpp / 원격 서버 | 각 주소 | OpenAI 호환이면 전부 동작 |

### CLI 백엔드 (구독 CLI 모델)

전체 프롬프트를 stdin으로 넘기고 stdout을 결과로 사용합니다. `cli_command`를
비우면 프리셋 명령이 쓰입니다:

| backend | 기본 명령 | 비고 |
|---|---|---|
| `claude_cli` | `claude -p --output-format text` | API 키 없이 구독으로 처리 |
| `gemini_cli` | `gemini -p` | 구글 계정/구독으로 처리 |
| `codex_cli` | `codex exec` | |
| `custom_cli` | (직접 입력 필수) | stdin→stdout 규약만 지키면 어떤 CLI든 가능 |

CLI 방식은 프로세스 기동 오버헤드 때문에 HTTP 서버 방식보다 호출당 수 초 느리고,
이미지 전달은 지원하지 않습니다(텍스트 전용).

## 팁

- `seed`는 캐시 무효화용입니다 — 같은 입력으로 다른 변주를 뽑고 싶으면 seed만 바꾸세요
  (기본값이 생성마다 자동 변경).
- Ref2VA에서 인물 사진을 `images`에 연결하면 `<Picture 1>`이 정체성 앵커가 됩니다.
  같은 이미지를 H3 레퍼런스 입력에도 연결하세요.
- SFW 모드는 전연령 유지 지시가, NSFW 모드는 무검열 지시가 시스템 프롬프트에 들어갑니다.
  실제 수위는 사용하는 LLM의 정책을 따릅니다.

## 크레딧

- H3 프롬프트 가이드는 [teskor-hub/minimax-h3-skill](https://github.com/teskor-hub/minimax-h3-skill)
  (MIT, © 2026 teskor) 기반이며, [MiniMax 공식 프롬프트 작성 가이드](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/docs) 규격을 따릅니다.
