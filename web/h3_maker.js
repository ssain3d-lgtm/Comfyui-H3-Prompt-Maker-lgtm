/**
 * Draws the H3 Prompt Maker (UI) node and hosts the overlay.
 *
 * The node's three widgets (state / llm / result) are written by the overlay,
 * never typed, so they are hidden and replaced by two buttons. The overlay is
 * the web app's own bundle in an iframe; everything crossing that boundary goes
 * through postMessage with a namespaced `source`, because ComfyUI's frontend
 * and every other extension share this window.
 */
import { app } from "../../scripts/app.js";

const NODE = "H3PromptMakerUI";
const PREFIX = "/h3_prompt_maker";
const HIDDEN = ["state", "llm", "result"];

const DEFAULT_PRESETS = {
  base: {
    lmstudio: "http://127.0.0.1:1234/v1",
    ollama: "http://127.0.0.1:11434/v1",
    llamacpp: "http://127.0.0.1:8080/v1",
    vllm: "http://127.0.0.1:8000/v1",
  },
  cli: {
    claude_cli: "claude -p --output-format text",
    gemini_cli: "gemini -p",
    codex_cli: "codex exec",
  },
};

const DEFAULT_LLM = {
  backend: "lmstudio",
  base_url: "",
  model: "",
  api_key: "",
  cli_command: "",
  server_model: "(auto)",
  temperature: 0.7,
  /** Epoch ms of the last successful 연결 확인, 0 when never checked. */
  verifiedAt: 0,
};

const parse = (raw, fallback) => {
  try { const v = JSON.parse(raw); return v && typeof v === "object" ? v : fallback; }
  catch { return fallback; }
};

const widget = (node, name) => node.widgets?.find((w) => w.name === name);
const readJson = (node, name, fallback) => parse(widget(node, name)?.value ?? "", fallback);
const writeJson = (node, name, value) => {
  const w = widget(node, name);
  if (w) w.value = JSON.stringify(value);
};

/** Room reserved under the widgets for the two status lines drawn by hand. */
const FOOTER_H = 40;

/**
 * Put air under a widget without adding one.
 *
 * The first attempt at this inserted a spacer widget — and ComfyUI backs a
 * "text" widget with a real DOM input, so giving it a positive height rendered
 * an empty textbox, cursor and all, between the buttons. Nothing is added here
 * instead: LiteGraph draws a button at its own fixed height but advances to the
 * next widget by whatever computeSize reports, so an inflated height leaves a
 * gap that has no element in it and cannot render anything.
 */
const padBelow = (widget, gap) => {
  // globalThis, not a bare LiteGraph: an undeclared identifier throws a
  // ReferenceError that optional chaining does not catch, which would take out
  // node creation entirely if ComfyUI ever stopped exposing it globally.
  const base = globalThis.LiteGraph?.NODE_WIDGET_HEIGHT ?? 20;
  widget.computeSize = (width) => [width, base + gap];
  return widget;
};

const resize = (node) => {
  const [w, h] = node.computeSize();
  node.size[0] = Math.max(w, 290);
  node.size[1] = h + FOOTER_H;
  node.setDirtyCanvas(true, true);
};

/** One line describing where generation will go, for the node face. */
const describeConn = (llm) => {
  const cfg = { ...DEFAULT_LLM, ...(llm || {}) };
  if (!llm || Object.keys(llm).length === 0) return { text: "모델 미설정", ok: false };
  const model = cfg.server_model && cfg.server_model !== "(auto)" ? cfg.server_model : (cfg.model || "(auto)");
  const verified = cfg.verifiedAt ? "● " : "○ ";
  return { text: `${verified}${cfg.backend} · ${model}`, ok: Boolean(cfg.verifiedAt) };
};

/**
 * A widget the user must not edit by hand still has to serialize, so it keeps
 * its value and loses only its drawing and its height.
 *
 * Zeroing computeSize is enough for a canvas-drawn widget but NOT for a
 * DOM-backed one: ComfyUI positions those elements itself and only skips the
 * type names it knows, so a custom type left the element on screen. The node no
 * longer asks for multiline (see nodes.py), but a workflow saved against the
 * older node still carries one, so the element is hidden explicitly here.
 */
const hideWidget = (w) => {
  w.type = "converted-widget";   // the name ComfyUI's own layout skips
  w.computeSize = () => [0, -4];
  w.computedHeight = 0;
  w.draw = () => {};
  w.onDrawBackground = () => {};
  w.options = { ...(w.options || {}), hidden: true };
  const el = w.element || w.inputEl;
  if (el && el.style) { el.style.display = "none"; el.hidden = true; }
};

// ---------------------------------------------------------------------------
// Overlay
// ---------------------------------------------------------------------------

let overlay = null;   // reused across opens so attachments survive a close
let overlayNode = null;

const ensureOverlay = () => {
  if (overlay) return overlay;

  const root = document.createElement("div");
  root.id = "h3-maker-overlay";
  Object.assign(root.style, {
    position: "fixed", inset: "0", zIndex: "1200", display: "none",
    background: "rgba(0,0,0,0.72)", backdropFilter: "blur(2px)",
  });

  const shell = document.createElement("div");
  Object.assign(shell.style, {
    position: "absolute", inset: "2.5vh 2.5vw", display: "flex", flexDirection: "column",
    borderRadius: "14px", overflow: "hidden", border: "1px solid #374151",
    boxShadow: "0 24px 64px rgba(0,0,0,0.6)", background: "#0d1117",
  });

  const bar = document.createElement("div");
  Object.assign(bar.style, {
    height: "38px", flex: "0 0 38px", display: "flex", alignItems: "center",
    justifyContent: "space-between", padding: "0 12px", background: "#161b22",
    borderBottom: "1px solid #30363d", color: "#c9d1d9",
    font: "600 12px/1 ui-sans-serif, system-ui, sans-serif",
  });
  const title = document.createElement("span");
  title.textContent = "🎬 MiniMax H3 Prompt Maker";
  const close = document.createElement("button");
  close.textContent = "✕ 닫기";
  Object.assign(close.style, {
    background: "transparent", border: "1px solid #30363d", borderRadius: "6px",
    color: "#c9d1d9", cursor: "pointer", padding: "4px 10px", font: "600 11px/1 inherit",
  });
  close.onclick = () => hideOverlay();
  bar.append(title, close);

  const frame = document.createElement("iframe");
  // Trailing slash matters: the bundle's asset links are relative to it.
  frame.src = `${PREFIX}/app/`;
  Object.assign(frame.style, { flex: "1 1 auto", width: "100%", border: "0", background: "#0d1117" });

  shell.append(bar, frame);
  root.append(shell);
  document.body.append(root);

  // Escape closes, but only while the overlay is the thing on screen — otherwise
  // it would swallow the key from ComfyUI's own dialogs.
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && root.style.display !== "none") { e.stopPropagation(); hideOverlay(); }
  }, true);

  overlay = { root, frame };
  return overlay;
};

const hideOverlay = () => {
  if (overlay) overlay.root.style.display = "none";
  overlayNode = null;
};

const sendToOverlay = (type, payload) => {
  overlay?.frame?.contentWindow?.postMessage(
    { source: "h3-prompt-maker-host", type, payload }, "*");
};

const openOverlay = (node) => {
  const { root } = ensureOverlay();
  overlayNode = node;
  root.style.display = "block";
  // The iframe may already be loaded from a previous open; push current values
  // either way — the app asks for them again with a `ready` message on first load.
  pushNodeState(node);
};

const pushNodeState = (node) => {
  sendToOverlay("llm", { ...DEFAULT_LLM, ...readJson(node, "llm", {}) });
  sendToOverlay("state", readJson(node, "state", {}));
};

window.addEventListener("message", (e) => {
  const d = e.data;
  if (!d || d.source !== "h3-prompt-maker") return;
  const node = overlayNode;
  if (!node) return;

  if (d.type === "ready") { pushNodeState(node); return; }
  if (d.type === "state") { writeJson(node, "state", d.payload || {}); return; }
  if (d.type === "close") { hideOverlay(); return; }
  if (d.type === "apply") {
    writeJson(node, "result", d.payload || {});
    node.h3Status = summarize(d.payload);
    node.setDirtyCanvas(true, true);
  }
});

const summarize = (r) => {
  if (!r || !r.prompt) return "적용된 프롬프트 없음";
  const n = r.segmentCount || 1;
  return `✓ ${r.lengthFrames || "?"}f` + (n > 1 ? ` · 세그먼트 ${n}개` : "") +
         ` · ${String(r.prompt).length.toLocaleString()}자`;
};

// ---------------------------------------------------------------------------
// Settings dialog — the only thing the node itself configures
// ---------------------------------------------------------------------------

const el = (tag, style, props = {}) => {
  const n = Object.assign(document.createElement(tag), props);
  Object.assign(n.style, style);
  return n;
};

const FIELD_STYLE = {
  background: "#0d1117", border: "1px solid #30363d", borderRadius: "6px",
  color: "#e6edf3", padding: "5px 8px", font: "inherit", width: "100%",
};

const button = (text, kind) => {
  const b = el("button", {
    padding: "6px 14px", borderRadius: "6px", cursor: "pointer",
    font: "600 12px/1 inherit", whiteSpace: "nowrap",
    border: kind === "primary" ? "0" : "1px solid #30363d",
    background: kind === "primary" ? "#4f46e5" : "transparent",
    color: kind === "primary" ? "#fff" : "#c9d1d9",
  }, { textContent: text, type: "button" });
  return b;
};

const openSettings = async (node) => {
  const cfg = { ...DEFAULT_LLM, ...readJson(node, "llm", {}) };
  let meta = { backends: [...Object.keys(DEFAULT_PRESETS.base), "openai_compat", "claude_cli", "custom_cli"],
               preset_base_urls: DEFAULT_PRESETS.base, preset_cli_commands: DEFAULT_PRESETS.cli, models: [] };
  try {
    const res = await fetch(`${PREFIX}/api/backends`);
    if (res.ok) meta = await res.json();
  } catch { /* offline: the presets still let the user type an address */ }

  const back = el("div", {
    position: "fixed", inset: "0", zIndex: "1300", display: "flex",
    alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.6)",
  });
  const box = el("div", {
    width: "min(520px, 94vw)", background: "#161b22", border: "1px solid #30363d",
    borderRadius: "12px", padding: "18px", color: "#c9d1d9",
    font: "13px/1.5 ui-sans-serif, system-ui, sans-serif",
    display: "flex", flexDirection: "column", gap: "10px",
  });
  box.append(el("div", { font: "700 14px/1 inherit" }, { textContent: "⚙️ 모델 연결" }));

  const inputs = {};
  const rows = {};
  const row = (key, label, node2) => {
    const r = el("label", { display: "grid", gridTemplateColumns: "96px 1fr", alignItems: "center", gap: "8px" });
    r.append(el("span", { fontSize: "12px", color: "#8b949e" }, { textContent: label }), node2);
    rows[key] = r;
    box.append(r);
    return r;
  };

  inputs.backend = el("select", FIELD_STYLE);
  for (const b of meta.backends) inputs.backend.append(new Option(b, b));
  inputs.backend.value = cfg.backend;
  row("backend", "백엔드", inputs.backend);

  inputs.base_url = el("input", FIELD_STYLE, { type: "text", value: cfg.base_url });
  row("base_url", "base_url", inputs.base_url);

  inputs.api_key = el("input", FIELD_STYLE, { type: "password", value: cfg.api_key });
  row("api_key", "API 키", inputs.api_key);

  inputs.cli_command = el("input", FIELD_STYLE, { type: "text", value: cfg.cli_command });
  row("cli_command", "CLI 명령", inputs.cli_command);

  // --- connect ---------------------------------------------------------
  const connectRow = el("div", { display: "flex", alignItems: "center", gap: "10px", marginTop: "2px" });
  const connectBtn = button("🔌 연결 확인", "primary");
  const statusText = el("span", { fontSize: "12px", color: "#8b949e", flex: "1", minWidth: "0" },
                        { textContent: cfg.verifiedAt ? "● 이전에 연결 확인됨" : "아직 확인하지 않음" });
  if (cfg.verifiedAt) statusText.style.color = "#34d399";
  connectRow.append(connectBtn, statusText);
  box.append(connectRow);

  // --- model, populated by connecting -----------------------------------
  const modelWrap = el("div", { display: "flex", gap: "6px" });
  inputs.server_model = el("select", FIELD_STYLE);
  const loadBtn = button("모델 로드", "ghost");
  loadBtn.disabled = true;
  loadBtn.style.opacity = "0.5";
  modelWrap.append(inputs.server_model, loadBtn);
  row("server_model", "모델", modelWrap);

  const fillModels = (models) => {
    inputs.server_model.textContent = "";
    inputs.server_model.append(new Option("(auto)", "(auto)"));
    for (const m of models) inputs.server_model.append(new Option(m, m));
    const saved = cfg.server_model && cfg.server_model !== "(auto)" ? cfg.server_model : cfg.model;
    if (saved && !models.includes(saved)) inputs.server_model.append(new Option(`${saved} (저장된 값)`, saved));
    inputs.server_model.value = saved || "(auto)";
  };
  fillModels(meta.models || []);

  inputs.temperature = el("input", FIELD_STYLE, { type: "number", step: "0.05", min: "0", max: "2", value: cfg.temperature });
  row("temperature", "temperature", inputs.temperature);

  // Only show the fields the chosen backend actually uses — an HTTP server has
  // no CLI command, and a CLI backend has no address or model list.
  const applyBackend = (fillPreset) => {
    const b = inputs.backend.value;
    const isCli = b.endsWith("_cli");
    if (fillPreset) {
      const url = meta.preset_base_urls?.[b];
      if (url) inputs.base_url.value = url;
      const cmd = meta.preset_cli_commands?.[b];
      if (cmd) inputs.cli_command.value = cmd;
    }
    rows.base_url.style.display = isCli ? "none" : "grid";
    rows.api_key.style.display = isCli ? "none" : "grid";
    rows.cli_command.style.display = isCli ? "grid" : "none";
    rows.server_model.style.display = isCli ? "none" : "grid";
    setUnverified("백엔드가 바뀌었습니다 — 연결을 다시 확인하세요.");
  };

  let verified = Boolean(cfg.verifiedAt);
  const setUnverified = (why) => {
    verified = false;
    loadBtn.disabled = true;
    loadBtn.style.opacity = "0.5";
    statusText.style.color = "#8b949e";
    statusText.textContent = why;
  };
  for (const key of ["base_url", "api_key", "cli_command"]) {
    inputs[key].addEventListener("input", () => setUnverified("설정이 바뀌었습니다 — 연결을 다시 확인하세요."));
  }
  inputs.backend.onchange = () => applyBackend(true);

  connectBtn.onclick = async () => {
    connectBtn.disabled = true;
    statusText.style.color = "#8b949e";
    statusText.textContent = "연결 확인 중…";
    try {
      const res = await fetch(`${PREFIX}/api/probe`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          backend: inputs.backend.value, base_url: inputs.base_url.value,
          api_key: inputs.api_key.value, cli_command: inputs.cli_command.value,
        }),
      });
      const r = await res.json();
      if (r.ok) {
        verified = true;
        statusText.style.color = "#34d399";
        statusText.textContent = `● 연결됨 — ${r.detail}`;
        if (r.kind === "http") {
          fillModels(r.models || []);
          loadBtn.disabled = false;
          loadBtn.style.opacity = "1";
        }
      } else {
        setUnverified("");
        statusText.style.color = "#f87171";
        statusText.textContent = `✗ ${r.detail}`;
      }
    } catch (e) {
      setUnverified("");
      statusText.style.color = "#f87171";
      statusText.textContent = `✗ 확인 실패: ${e.message}`;
    } finally {
      connectBtn.disabled = false;
    }
  };

  loadBtn.onclick = async () => {
    const model = inputs.server_model.value;
    if (!model || model === "(auto)") {
      statusText.style.color = "#fb923c";
      statusText.textContent = "로드할 모델을 목록에서 고르세요 ((auto)는 로드 대상이 없습니다).";
      return;
    }
    loadBtn.disabled = true;
    statusText.style.color = "#8b949e";
    // Loading a large model off disk is slow; saying so beats a frozen dialog.
    statusText.textContent = `${model} 로드 중… (모델 크기에 따라 수십 초 걸릴 수 있습니다)`;
    try {
      const res = await fetch(`${PREFIX}/api/load-model`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          backend: inputs.backend.value, base_url: inputs.base_url.value,
          api_key: inputs.api_key.value, model,
        }),
      });
      const r = await res.json();
      statusText.style.color = r.ok ? "#34d399" : "#f87171";
      statusText.textContent = (r.ok ? "● " : "✗ ") + r.detail;
    } catch (e) {
      statusText.style.color = "#f87171";
      statusText.textContent = `✗ 로드 실패: ${e.message}`;
    } finally {
      loadBtn.disabled = false;
    }
  };

  applyBackend(false);

  const hint = el("p", { margin: "2px 0 0", fontSize: "11px", color: "#8b949e", lineHeight: "1.5" });
  hint.innerHTML = "위젯에 입력한 키는 <b>워크플로우 JSON과 그 워크플로우로 만든 PNG 메타데이터에 저장</b>됩니다. " +
                   "공유할 계획이면 비워 두고 환경변수를 쓰세요.";
  box.append(hint);

  const footer = el("div", { display: "flex", justifyContent: "flex-end", gap: "8px", marginTop: "4px" });
  const cancel = button("취소", "ghost");
  const save = button("저장", "primary");
  cancel.onclick = () => back.remove();
  save.onclick = () => {
    const next = {
      backend: inputs.backend.value,
      base_url: inputs.base_url.value,
      api_key: inputs.api_key.value,
      cli_command: inputs.cli_command.value,
      server_model: inputs.server_model.value,
      model: inputs.server_model.value === "(auto)" ? "" : inputs.server_model.value,
      temperature: Number(inputs.temperature.value) || 0.7,
      // Recorded so the node face can distinguish a checked configuration from
      // one that was merely typed in and saved.
      verifiedAt: verified ? Date.now() : 0,
    };
    writeJson(node, "llm", next);
    node.h3Conn = describeConn(next);
    if (overlayNode === node) sendToOverlay("llm", next);
    node.setDirtyCanvas(true, true);
    back.remove();
  };
  footer.append(cancel, save);
  box.append(footer);
  back.append(box);
  back.onclick = (e) => { if (e.target === back) back.remove(); };
  document.body.append(back);
  connectBtn.focus();
};

// ---------------------------------------------------------------------------

app.registerExtension({
  name: "h3.prompt.maker.ui",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      for (const name of HIDDEN) {
        const w = widget(this, name);
        if (w) hideWidget(w);
      }
      // LiteGraph stacks button widgets flush, so the primary action and the
      // settings button read as one control. The gap goes on the first button
      // itself — see padBelow.
      const open = this.addWidget("button", "🎬 프롬프트 메이커 열기", null, () => openOverlay(this));
      padBelow(open, 8);
      this.addWidget("button", "⚙️ 모델 연결", null, () => openSettings(this));

      this.h3Status = summarize(readJson(this, "result", {}));
      this.h3Conn = describeConn(readJson(this, "llm", {}));
      resize(this);
      return r;
    };

    // A node whose whole state is hidden needs to say what it is holding, or a
    // reopened workflow looks empty when it is not.
    const onDraw = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      onDraw?.apply(this, arguments);
      if (this.flags?.collapsed) return;
      ctx.save();
      ctx.font = "11px ui-sans-serif, system-ui, sans-serif";

      // Connection first, then what is loaded: the order you need them in.
      const conn = this.h3Conn || { text: "모델 미설정", ok: false };
      ctx.fillStyle = conn.ok ? "#34d399" : "#8b949e";
      ctx.fillText(conn.text, 12, this.size[1] - FOOTER_H + 14);

      const status = this.h3Status || "적용된 프롬프트 없음";
      ctx.fillStyle = status.startsWith("✓") ? "#34d399" : "#8b949e";
      ctx.fillText(status, 12, this.size[1] - FOOTER_H + 31);
      ctx.restore();
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      this.h3Status = summarize(readJson(this, "result", {}));
      this.h3Conn = describeConn(readJson(this, "llm", {}));
      resize(this);
      return r;
    };
  },
});
