# ComfyUI H3 Prompt Maker

MiniMax H3 (비디오+오디오 옴니모달 모델) 전용 **프롬프트 생성 커스텀 노드**입니다.
장면 요청(한국어 가능)과 참조 이미지를 넣으면 LLM이 H3 공식 포맷
(T2VA / I2VA / FL2VA / L2VA / Ref2VA)의 영어 프롬프트를 설계하고,
결과를 **같은 워크플로우의 H3 노드에 바로 와이어로 연결**할 수 있습니다.

[H3 Prompt Maker 웹앱](https://github.com/ssain3d-lgtm/minimax-h3-prompt-maker-google-studio-ai-v3)과
동일한 시스템 프롬프트를 사용합니다 (prompts.ts에서 그대로 추출).

## 설치

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ssain3d-lgtm/Comfyui-H3-Prompt-Maker-lgtm-.git
# 추가 의존성 없음 (표준 라이브러리 + ComfyUI 기본 패키지만 사용)
```

ComfyUI 재시작 후 노드 검색에서 `H3 Prompt Maker` 카테고리를 찾으세요.
UI 방식을 원하면 **MiniMax H3 Prompt Maker (UI) 🖥️** 를, 그래프에 직접 와이어링하려면
**Prompt Architect 🎬** 를 추가하세요.

## 노드

### 🖥️ MiniMax H3 Prompt Maker (UI)
**웹앱 UI를 ComfyUI 위에 그대로 띄우는 노드.** 노드에는 버튼 두 개뿐입니다.

| 버튼 | 하는 일 |
|---|---|
| 🎬 프롬프트 메이커 열기 | 웹앱 화면 전체를 오버레이로 엽니다 — 세부 모드·길이·SFW/NSFW·장면 요청·대사·목소리·카메라·리메이크 모드·참조 첨부·오디오 트림·히스토리 전부 동일 |
| ⚙️ 모델 연결 | 백엔드 / 주소 / 모델 / 키 / temperature / max_tokens. **이 노드가 직접 설정하는 것은 이것뿐입니다** |

**⚙️ 모델 연결 다이얼로그**

1. 백엔드를 고르면 표준 주소가 자동으로 채워지고, 그 백엔드가 쓰지 않는 칸은 숨겨집니다
   (HTTP 서버에는 CLI 명령이, CLI 백엔드에는 주소·모델 목록이 없습니다).
2. **🔌 연결 확인** — 실제로 서버에 닿아 봅니다. 성공하면 **`● 연결됨 — 모델 N개 확인됨`**,
   실패하면 빨간 글씨로 이유가 그대로 나옵니다 (주소 오타 / 서버 꺼짐 / 키 오류).
   생성할 때가 아니라 **여기서** 알 수 있다는 게 요점입니다.
3. 확인에 성공하면 **모델 목록이 실제 서버 응답으로 채워집니다.**
4. **모델 로드** — 고른 모델을 서버 메모리에 미리 올립니다. LM Studio·Ollama·vLLM은 첫 요청
   때 모델을 로드하는데, 그 지연이 길어서 첫 생성이 멈춘 것처럼 보입니다. 미리 눌러 두세요.
5. 주소·키를 다시 건드리면 확인 상태가 해제됩니다 — 확인하지 않은 설정이 확인된 것처럼
   보이지 않게 하기 위함입니다.

**max_tokens (기본 60000)** — 응답 상한입니다. 추론 모델(Qwen3 등)은 답하기 전에 이 예산을
**생각하는 데** 씁니다. 낮으면 사고 블록이 예산을 다 먹고 본문 없이 한 줄만 돌아옵니다.
느린 하드웨어에서 응답이 너무 오래 걸리면 줄이고, 긴 프롬프트가 잘리면 올리세요.

Qwen3 계열에서 계속 한 줄만 나오면 장면 요청 끝에 `/no_think`를 붙여 사고를 꺼도 됩니다.
(사고 블록 안에 본문을 다 써버린 경우는 앱이 알아서 그 안의 프롬프트를 꺼내 씁니다.)

노드 얼굴에 연결 상태가 한 줄로 남습니다: `● lmstudio · qwen3-14b-instruct`
(● = 연결 확인됨, ○ = 저장만 됨).

오버레이에서 생성한 뒤 **'이 노드에 적용'** 을 누르면 결과가 노드 출력
(`prompt` / `length_frames` / `korean_summary` / `all_segments` / `segment_count`)으로 나갑니다.
Queue를 눌러도 LLM을 다시 호출하지 않습니다 — 눈으로 확인하고 승인한 프롬프트만 렌더에 들어갑니다.
아직 적용한 게 없으면 실행이 그 사실을 말하며 멈춥니다.

알아두실 점:

- **참조 미디어는 워크플로우에 저장되지 않습니다.** ComfyUI 워크플로우는 그 워크플로우로 만든
  모든 PNG 메타데이터에 통째로 들어가므로, 10MB 오디오를 넣으면 출력 이미지마다 따라다닙니다.
  워크플로우에는 장면·설정 텍스트만 들어갑니다.
- 대신 **첨부는 브라우저 IndexedDB에 히스토리별로 남습니다.** 오버레이 하단 히스토리 카드를
  누르면 그때 쓴 이미지·비디오·오디오와 역할 메모가 그대로 돌아옵니다 (ComfyUI를 껐다 켜도 유지).
  총량 300MB를 넘으면 오래된 것부터 비워집니다.
- **마스크 → AI 인페인팅은 없습니다.** Gemini 이미지 모델 전용 기능이라 로컬 LLM으로 대체할 수 없습니다.
  ComfyUI의 인페인트 노드를 그래프에서 쓰세요.
- 오버레이는 **완전 오프라인**입니다. Tailwind를 CDN이 아니라 빌드된 CSS로 싣습니다.

아래 두 노드는 그래프에 직접 와이어링하고 싶을 때 쓰는 위젯 방식입니다. UI 노드와 같은
시스템 프롬프트·같은 출력 5개를 냅니다.

### 🎬 MiniMax H3 Prompt Architect
장면 요청 → 완성된 H3 프롬프트.

| 출력 | 용도 |
|---|---|
| `prompt` (STRING) | H3 텍스트 인코딩 노드에 연결 — **이번 렌더 한 개분**(분할 생성 시 세그먼트 1) |
| `length_frames` (INT) | H3 length/frames 입력에 연결 — `prompt`와 항상 짝이 맞음 |
| `korean_summary` (STRING) | 한국어 요약 (모델이 제공한 경우) |
| `all_segments` (STRING) | 분할 생성 시 전체 시퀀스(헤더 포함). 단일 렌더면 `prompt`와 동일 |
| `segment_count` (INT) | 세그먼트 개수 (단일 렌더 = 1) |

코드펜스·`length` 라인·추론(`<think>`) 블록은 자동으로 제거되어 `prompt`는 붙여넣기 가능한 상태로 나옵니다.

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
- `api_key`를 비워 두면 환경변수(`OPENAI_API_KEY` / `OPENROUTER_API_KEY` / `GEMINI_API_KEY` / `H3_LLM_API_KEY`)를 사용합니다.
  **위젯에 직접 입력한 키는 워크플로우 JSON과 그 워크플로우로 만든 모든 PNG 메타데이터에 저장되므로**, 공유할 계획이면 환경변수를 쓰세요.

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

명령은 셸 없이 실행됩니다(`shlex.split` + `shell=False`). 공유받은 워크플로우의
`cli_command`에 담긴 파이프·`;`·`$()` 체이닝이 실행되지 않도록 하기 위함입니다.

## 웹앱과 동기화

이 저장소는 웹앱에서 두 가지를 그대로 가져옵니다 — 시스템 프롬프트(`h3_prompts.py`)와
오버레이 UI(`web/app/`). 둘 다 **생성물이고 커밋되어 있습니다**(사용자가 npm을 돌릴 필요 없음).
웹앱이 바뀌면 손으로 고치지 말고 재생성하세요:

```bash
python3 tools/sync_app.py /path/to/minimax-h3-prompt-maker-google-studio-ai-v3
```

프롬프트 추출 → 오버레이 빌드 → `web/app/` 교체를 한 번에 하고, 번들에 CDN 참조가
남아 있으면 실패합니다.

## 테스트

```bash
python3 tests/test_parse.py     # 출력 파서 회귀 (INT 출력이 H3 샘플러로 직결됨)
python3 tests/test_routes.py    # 오버레이 HTTP 계층 — 경로 탈출 차단, 요청 번역
python3 tests/test_system_prompt.py  # H3 시스템 프롬프트가 실제 요청에 실리는지 (가짜 LLM 서버로 본문 캡처)
python3 tools/check_bundle.py   # 커밋된 번들이 서빙 가능하고 외부 참조가 없는지
```

`_parse_llm_output`은 이 노드팩에서 유일하게 순수하면서 잘 깨지는 함수이고,
INT 출력이 H3 샘플러로 직결되므로 출력 형태별 회귀 테스트를 고정해 두었습니다.
라우트 쪽은 ComfyUI 서버에 얹히는데 ComfyUI에는 자체 인증이 없으므로,
호출자가 준 문자열이 파일 시스템에 닿는 유일한 지점(정적 파일 핸들러)을 특히 고정해 두었습니다.

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
