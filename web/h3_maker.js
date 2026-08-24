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

const DEFAULT_LLM = {
  backend: "lmstudio",
  base_url: "",
  model: "",
  api_key: "",
  cli_command: "",
  server_model: "(auto)",
  temperature: 0.7,
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

/** A widget the user must not edit by hand still has to serialize, so it keeps
 *  its value and loses only its drawing and its height. */
const hideWidget = (w) => {
  w.type = "h3hidden";
  w.computeSize = () => [0, -4];
  w.draw = () => {};
  w.onDrawBackground = () => {};
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

const FIELDS = [
  ["backend", "백엔드", "select"],
  ["server_model", "감지된 모델", "select-models"],
  ["model", "모델 이름", "text"],
  ["base_url", "base_url", "text"],
  ["api_key", "API 키", "password"],
  ["cli_command", "CLI 명령", "text"],
  ["temperature", "temperature", "number"],
];

const openSettings = async (node) => {
  const cfg = { ...DEFAULT_LLM, ...readJson(node, "llm", {}) };
  let meta = { backends: ["lmstudio"], preset_base_urls: {}, preset_cli_commands: {}, models: [] };
  try {
    const res = await fetch(`${PREFIX}/api/backends`);
    if (res.ok) meta = await res.json();
  } catch { /* offline: the presets below still let the user type an address */ }

  const back = document.createElement("div");
  Object.assign(back.style, {
    position: "fixed", inset: "0", zIndex: "1300", display: "flex",
    alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.6)",
  });
  const box = document.createElement("div");
  Object.assign(box.style, {
    width: "min(460px, 92vw)", background: "#161b22", border: "1px solid #30363d",
    borderRadius: "12px", padding: "18px", color: "#c9d1d9",
    font: "13px/1.5 ui-sans-serif, system-ui, sans-serif",
    display: "flex", flexDirection: "column", gap: "10px",
  });
  const h = document.createElement("div");
  h.textContent = "⚙️ 모델 연결";
  h.style.font = "700 14px/1 inherit";
  box.append(h);

  const inputs = {};
  for (const [key, label, kind] of FIELDS) {
    const row = document.createElement("label");
    Object.assign(row.style, { display: "grid", gridTemplateColumns: "108px 1fr", alignItems: "center", gap: "8px" });
    const name = document.createElement("span");
    name.textContent = label;
    name.style.fontSize = "12px";
    name.style.color = "#8b949e";

    let field;
    if (kind === "select" || kind === "select-models") {
      field = document.createElement("select");
      const opts = kind === "select" ? meta.backends : ["(auto)", ...(meta.models || [])];
      for (const o of opts) {
        const el = document.createElement("option");
        el.value = o; el.textContent = o;
        field.append(el);
      }
      if (!opts.includes(cfg[key])) {
        const el = document.createElement("option");
        el.value = cfg[key]; el.textContent = cfg[key] + " (저장된 값)";
        field.append(el);
      }
    } else {
      field = document.createElement("input");
      field.type = kind === "password" ? "password" : kind === "number" ? "number" : "text";
      if (kind === "number") { field.step = "0.05"; field.min = "0"; field.max = "2"; }
    }
    field.value = cfg[key] ?? "";
    Object.assign(field.style, {
      background: "#0d1117", border: "1px solid #30363d", borderRadius: "6px",
      color: "#e6edf3", padding: "5px 8px", font: "inherit", width: "100%",
    });
    inputs[key] = field;
    row.append(name, field);
    box.append(row);
  }

  // Picking a preset backend should fill the address it implies, but never
  // overwrite an address the user typed for openai_compat.
  inputs.backend.onchange = () => {
    const b = inputs.backend.value;
    const url = meta.preset_base_urls?.[b];
    if (url) inputs.base_url.value = url;
    const cmd = meta.preset_cli_commands?.[b];
    if (cmd) inputs.cli_command.value = cmd;
  };

  const hint = document.createElement("p");
  hint.innerHTML = "위젯에 입력한 키는 <b>워크플로우 JSON과 그 워크플로우로 만든 PNG 메타데이터에 저장</b>됩니다. " +
                   "공유할 계획이면 비워 두고 환경변수를 쓰세요.";
  Object.assign(hint.style, { margin: "2px 0 0", fontSize: "11px", color: "#8b949e", lineHeight: "1.5" });
  box.append(hint);

  const row = document.createElement("div");
  Object.assign(row.style, { display: "flex", justifyContent: "flex-end", gap: "8px", marginTop: "4px" });
  const mk = (text, primary) => {
    const b = document.createElement("button");
    b.textContent = text;
    Object.assign(b.style, {
      padding: "6px 14px", borderRadius: "6px", cursor: "pointer", font: "600 12px/1 inherit",
      border: primary ? "0" : "1px solid #30363d",
      background: primary ? "#4f46e5" : "transparent",
      color: primary ? "#fff" : "#c9d1d9",
    });
    return b;
  };
  const cancel = mk("취소", false);
  const save = mk("저장", true);
  cancel.onclick = () => back.remove();
  save.onclick = () => {
    const next = {};
    for (const [key] of FIELDS) next[key] = inputs[key].value;
    next.temperature = Number(next.temperature) || 0.7;
    writeJson(node, "llm", next);
    if (overlayNode === node) sendToOverlay("llm", next);
    node.setDirtyCanvas(true, true);
    back.remove();
  };
  row.append(cancel, save);
  box.append(row);
  back.append(box);
  back.onclick = (e) => { if (e.target === back) back.remove(); };
  document.body.append(back);
  inputs.backend.focus();
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
      this.addWidget("button", "🎬 프롬프트 메이커 열기", null, () => openOverlay(this));
      this.addWidget("button", "⚙️ 모델 연결", null, () => openSettings(this));
      this.h3Status = summarize(readJson(this, "result", {}));
      this.size = this.computeSize();
      this.size[0] = Math.max(this.size[0], 260);
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
      ctx.fillStyle = this.h3Status?.startsWith("✓") ? "#34d399" : "#8b949e";
      ctx.fillText(this.h3Status || "적용된 프롬프트 없음", 12, this.size[1] - 8);
      ctx.restore();
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      this.h3Status = summarize(readJson(this, "result", {}));
      return r;
    };
  },
});
