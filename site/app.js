import { LocalNeuroEngine } from "./engine.js?v=20260302-09";

const els = {
  runtimePill: document.getElementById("runtime-pill"),
  precisionTag: document.getElementById("precision-tag"),
  messages: document.getElementById("messages"),
  form: document.getElementById("chat-form"),
  prompt: document.getElementById("prompt-input"),
  controlsDrawer: document.querySelector(".controls-drawer"),
  sendBtn: document.getElementById("send-btn"),
  temperature: document.getElementById("temperature"),
  topP: document.getElementById("top-p"),
  repetitionPenalty: document.getElementById("repetition-penalty"),
  maxNew: document.getElementById("max-new"),
  temperatureValue: document.getElementById("temperature-value"),
  topPValue: document.getElementById("top-p-value"),
  repetitionPenaltyValue: document.getElementById("repetition-penalty-value"),
  maxNewValue: document.getElementById("max-new-value"),
  historyMode: document.getElementById("history-mode"),
  thinkMode: document.getElementById("think-mode"),
  signalCanvas: document.getElementById("signal-canvas"),
  signalState: document.getElementById("signal-state"),
  startupOverlay: document.getElementById("startup-overlay"),
  startupTitle: document.getElementById("startup-title"),
  startupStage: document.getElementById("startup-stage"),
  startupSub: document.getElementById("startup-sub"),
  startupProgressTrack: document.getElementById("startup-progress-track"),
  startupProgressFill: document.getElementById("startup-progress-fill"),
  startupRetry: document.getElementById("startup-retry"),
  metrics: {
    tps: document.getElementById("m-tps"),
    ftl: document.getElementById("m-ftl"),
    out: document.getElementById("m-out"),
    in: document.getElementById("m-in"),
    mem: document.getElementById("m-mem"),
    ratio: document.getElementById("m-ratio"),
  },
  details: {
    name: document.getElementById("d-name"),
    arch: document.getElementById("d-arch"),
    quant: document.getElementById("d-quant"),
    path: document.getElementById("d-path"),
    data: document.getElementById("d-data"),
  },
};

const engine = new LocalNeuroEngine();
let busy = false;
let runtimeReady = false;
let activeAbortController = null;
// AR(3) coefficients for computational-load process: x_t = a1*x_{t-1} + a2*x_{t-2} + a3*x_{t-3} + eps_t
const LOAD_AR3_COEFFS = [2.2, -1.6, 0.36];
// Constant speed / sampling rate (samples per second).
const LOAD_SAMPLE_HZ = 60;
const LOAD_SAMPLE_INTERVAL_MS = 1000 / LOAD_SAMPLE_HZ;
const SIGNAL_DRAW_INTERVAL_MS = 33;
// Idle variance/amplitude (as Gaussian noise std dev).
const LOAD_IDLE_STD = 0.0005;
// Generation variance/amplitude (as Gaussian noise std dev).
const LOAD_BUSY_STD = 0.01;
// Transition speeds for idle->generation and generation->idle envelope.
const LOAD_TRANSITION_UP = 0.12;
const LOAD_TRANSITION_DOWN = 0.05;

const LOAD_Y_SCALE = 2.3;
let loadStdCurrent = LOAD_IDLE_STD;
let loadStdTarget = LOAD_IDLE_STD;
let loadAr1 = 0;
let loadAr2 = 0;
let loadAr3 = 0;
let loadAccumulatorMs = 0;
let loadLastTs = 0;
let loadLastDrawTs = 0;
let loadTrace = [];
const MAX_MESSAGES = 80;
const MAX_HISTORY_MESSAGES = 24;
const HISTORY_RECENT_MESSAGES = 8;
const HISTORY_SUMMARY_MAX_ITEMS = 10;
const HISTORY_SUMMARY_CHAR_CAP = 1400;
const THINK_TOKEN_CAP = 1028;
const THINK_TIME_LIMIT_MS = 60000;
const STREAM_RENDER_INTERVAL_MS = 120;
const THINK_ANSWER_SCAN_INTERVAL_MS = 180;
const MOBILE_LAYOUT_QUERY = "(max-width: 1020px)";
const conversationHistory = [];
const textMeasureCanvas = document.createElement("canvas");
const textMeasureCtx = textMeasureCanvas.getContext("2d");

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

let markdownConfigured = false;
let mathTypesetTimer = null;

function ensureMarkdownConfigured() {
  if (markdownConfigured) return;
  const md = globalThis.marked;
  if (md?.setOptions) {
    md.setOptions({
      gfm: true,
      breaks: true,
      mangle: false,
      headerIds: false,
    });
  }
  markdownConfigured = true;
}

function renderMarkdown(text) {
  const src = String(text || "").replace(/\r/g, "").trim();
  if (!src) return "";

  const md = globalThis.marked;
  const parseMd =
    typeof md?.parse === "function"
      ? (value) => md.parse(value)
      : typeof md === "function"
        ? (value) => md(value)
        : null;

  if (parseMd) {
    ensureMarkdownConfigured();
    let html = parseMd(src);
    if (globalThis.DOMPurify?.sanitize) {
      html = globalThis.DOMPurify.sanitize(html, {
        USE_PROFILES: { html: true },
      });
    }
    return html;
  }

  return `<p>${escapeHtml(src).replace(/\n/g, "<br>")}</p>`;
}

function queueMathTypeset(container) {
  const mj = globalThis.MathJax;
  if (!mj?.typesetPromise || !container) return;
  if (mathTypesetTimer) {
    clearTimeout(mathTypesetTimer);
  }
  mathTypesetTimer = setTimeout(() => {
    mj.typesetPromise([container]).catch(() => {});
  }, 120);
}

function setAssistantBody(frame, text, options = {}) {
  const content = String(text || "").trim();
  if (!content) {
    frame.body.textContent = "";
    return;
  }
  frame.body.innerHTML = renderMarkdown(content);
  if (options.typesetMath !== false) {
    queueMathTypeset(frame.body);
  }
}

function setStatus(text) {
  els.runtimePill.textContent = text;
}

function isMobileLayout() {
  return window.matchMedia(MOBILE_LAYOUT_QUERY).matches;
}

function syncControlsDrawerByViewport() {
  if (!els.controlsDrawer) return;
  els.controlsDrawer.open = !isMobileLayout();
}

function setStartupVisible(visible) {
  if (!els.startupOverlay) return;
  els.startupOverlay.classList.toggle("hidden", !visible);
}

function setStartupProgress(progress) {
  if (!els.startupProgressFill || !els.startupProgressTrack) return;
  if (!Number.isFinite(progress)) {
    els.startupProgressTrack.classList.remove("hidden");
    els.startupProgressFill.style.width = "15%";
    return;
  }
  const bounded = Math.max(0, Math.min(100, progress));
  els.startupProgressTrack.classList.remove("hidden");
  els.startupProgressFill.style.width = `${bounded}%`;
}

function updateStartupView({ title, stage, sub, progress, failed = false }) {
  if (els.startupTitle && title) els.startupTitle.textContent = String(title);
  if (els.startupStage && stage) els.startupStage.textContent = String(stage);
  if (els.startupSub && sub) els.startupSub.textContent = String(sub);
  setStartupProgress(progress);
  if (els.startupRetry) {
    els.startupRetry.classList.toggle("hidden", !failed);
  }
}

function formatMb(bytes) {
  const mb = Number(bytes) / (1024 * 1024);
  if (!Number.isFinite(mb) || mb <= 0) return "?";
  return mb.toFixed(mb < 100 ? 1 : 0);
}

function runtimeLabel(mode, phase = "active") {
  const m = String(mode || "");
  if (m.includes("error")) {
    return "Candle WASM CPU error";
  }
  if (m.startsWith("candle-wasm")) {
    return phase === "ready" ? "Candle WASM CPU ready" : "Candle WASM CPU active";
  }
  if (m.startsWith("unavailable") || m === "initializing") {
    return phase === "ready" ? "WASM runtime unavailable" : "Initializing WASM runtime";
  }
  return phase === "ready" ? "Runtime ready" : "Runtime active";
}

function scrollMessages() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

function pruneMessages() {
  while (els.messages.children.length > MAX_MESSAGES) {
    els.messages.removeChild(els.messages.firstElementChild);
  }
}

function addMessage(role, text) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const p = document.createElement("p");
  p.textContent = text;
  article.appendChild(p);
  els.messages.appendChild(article);
  pruneMessages();
  scrollMessages();
  return p;
}

function createAssistantFrame() {
  const article = document.createElement("article");
  article.className = "message assistant assistant-frame";

  const body = document.createElement("div");
  body.className = "assistant-body";
  body.textContent = "Thinking...";
  article.appendChild(body);

  const thinkingLive = document.createElement("section");
  thinkingLive.className = "thinking-live hidden";

  const thinkingLine = document.createElement("div");
  thinkingLine.className = "thinking-line thinking-wave";
  thinkingLine.textContent = "Thinking";
  thinkingLive.appendChild(thinkingLine);

  const thinkingPreview = document.createElement("div");
  thinkingPreview.className = "thinking-preview-line";
  thinkingLive.appendChild(thinkingPreview);

  const thinkingPreviewFull = document.createElement("pre");
  thinkingPreviewFull.className = "thinking-preview-full hidden";
  thinkingLive.appendChild(thinkingPreviewFull);

  const thinkingPreviewHint = document.createElement("div");
  thinkingPreviewHint.className = "thinking-preview-hint hidden";
  thinkingPreviewHint.textContent = "Click to expand";
  thinkingLive.appendChild(thinkingPreviewHint);

  article.appendChild(thinkingLive);

  const thought = document.createElement("section");
  thought.className = "thought-panel hidden";

  const thoughtHead = document.createElement("div");
  thoughtHead.className = "thought-head";

  const thoughtTitle = document.createElement("span");
  thoughtTitle.textContent = "Thinking Trace (Internal)";

  const thoughtToggle = document.createElement("button");
  thoughtToggle.type = "button";
  thoughtToggle.className = "thought-toggle-btn";
  thoughtToggle.textContent = "Show trace";

  thoughtHead.appendChild(thoughtTitle);
  thoughtHead.appendChild(thoughtToggle);
  thought.appendChild(thoughtHead);

  const thoughtRecent = document.createElement("pre");
  thoughtRecent.className = "thought-recent hidden";
  thought.appendChild(thoughtRecent);

  const thoughtFull = document.createElement("pre");
  thoughtFull.className = "thought-full hidden";
  thought.appendChild(thoughtFull);

  article.appendChild(thought);
  els.messages.appendChild(article);
  pruneMessages();
  scrollMessages();

  const frame = {
    article,
    body,
    thinkingLive,
    thinkingPreview,
    thinkingPreviewFull,
    thinkingPreviewHint,
    thought,
    thoughtRecent,
    thoughtFull,
    thoughtTitle,
    thoughtToggle,
    expanded: false,
    liveExpanded: false,
    liveCanExpand: false,
    liveThoughtText: "",
    previewMetrics: { width: 0, font: "" },
  };

  thoughtToggle.addEventListener("click", () => {
    frame.expanded = !frame.expanded;
    thoughtToggle.textContent = frame.expanded ? "Hide trace" : "Show trace";
    if (frame.expanded) {
      thoughtFull.classList.remove("hidden");
    } else {
      thoughtFull.classList.add("hidden");
    }
    scrollMessages();
  });

  thinkingLive.addEventListener("click", () => {
    if (!frame.liveCanExpand) return;
    frame.liveExpanded = !frame.liveExpanded;
    if (frame.liveExpanded) {
      frame.thinkingLive.classList.add("live-expanded");
      frame.thinkingPreview.classList.add("hidden");
      frame.thinkingPreviewFull.textContent = frame.liveThoughtText || "";
      frame.thinkingPreviewFull.classList.remove("hidden");
      frame.thinkingPreviewHint.textContent = "Click to collapse";
    } else {
      frame.thinkingLive.classList.remove("live-expanded");
      frame.thinkingPreview.classList.remove("hidden");
      frame.thinkingPreviewFull.classList.add("hidden");
      frame.thinkingPreviewHint.textContent = "Click to expand";
    }
    frame.thinkingPreviewHint.classList.remove("hidden");
    scrollMessages();
  });

  return frame;
}

function initThinkingPreviewSizing(frame) {
  if (!frame?.thinkingPreview) return;
  const preview = frame.thinkingPreview;
  const cs = window.getComputedStyle(preview);
  frame.previewMetrics.width = Math.max(0, preview.getBoundingClientRect().width - 2);
  frame.previewMetrics.font =
    cs.font && cs.font !== ""
      ? cs.font
      : `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
}

function extractThought(answerText, assumePrefilledOpenThink = false) {
  const raw = String(answerText || "");
  const segments = [];
  let remaining = raw;

  remaining = remaining.replace(/<(think|thinking)>([\s\S]*?)<\/\1>/gi, (_all, _tag, chunk) => {
    segments.push(String(chunk || ""));
    return "";
  });

  const openTail = remaining.match(/<(think|thinking)>([\s\S]*)$/i);
  if (openTail) {
    const idx = openTail.index ?? -1;
    if (idx >= 0) {
      const tail = String(openTail[2] || "");
      segments.push(tail);
      remaining = remaining.slice(0, idx);
    }
  }

  // Some runtimes prefill `<think>\n` in the assistant prompt (not emitted in output).
  // In that case, generated text may start with reasoning and only later emit `</think>`.
  // Treat text before the first closing tag as thought when enabled.
  if (assumePrefilledOpenThink && segments.length === 0) {
    const closeMatch = remaining.match(/<\/(think|thinking)>/i);
    if (closeMatch && typeof closeMatch.index === "number") {
      const closeStart = closeMatch.index;
      const closeEnd = closeStart + closeMatch[0].length;
      segments.push(remaining.slice(0, closeStart));
      remaining = remaining.slice(closeEnd);
    } else {
      const trimmed = remaining.trim();
      if (trimmed) {
        segments.push(remaining);
        remaining = "";
      }
    }
  }

  const answer = remaining
    .replace(/<\/?(think|thinking)>/gi, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  const thought = segments.join("\n").trim();
  return { thought, answer };
}

function recentThoughtSlice(text) {
  const src = String(text || "").trim();
  if (!src) return "";
  const lines = src.split("\n").filter((line) => line.trim().length > 0);
  const tail = lines.slice(-6).join("\n");
  if (tail.length <= 380) return tail;
  return `...${tail.slice(-380)}`;
}

function currentThoughtSegment(text) {
  const src = String(text || "").replace(/\r/g, "");
  if (!src) return "";
  const blocks = src
    .split(/\n\s*\n+/)
    .map((chunk) => chunk.replace(/\s+/g, " ").trim())
    .filter(Boolean);
  if (blocks.length > 0) {
    return blocks[blocks.length - 1];
  }
  return src.replace(/\s+/g, " ").trim();
}

function normalizeThoughtText(text) {
  return String(text || "").replace(/\r/g, "").replace(/\n{3,}/g, "\n\n").trim();
}

function truncateToSingleLineByWidth(text, el, metricsCache) {
  const src = String(text || "").replace(/\s+/g, " ").trim();
  if (!src || !el || !textMeasureCtx) {
    return { text: src, truncated: false, full: src };
  }

  const cachedWidth = Number(metricsCache?.width || 0);
  const width = cachedWidth > 0 ? cachedWidth : Math.max(0, el.clientWidth - 2);
  if (width <= 0) {
    return { text: src, truncated: false, full: src };
  }

  let font = String(metricsCache?.font || "");
  if (!font) {
    const cs = window.getComputedStyle(el);
    font =
      cs.font && cs.font !== ""
        ? cs.font
        : `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
  }

  if (metricsCache) {
    metricsCache.width = width;
    metricsCache.font = font;
  }
  textMeasureCtx.font = font;

  if (textMeasureCtx.measureText(src).width <= width) {
    return { text: src, truncated: false, full: src };
  }

  const ellipsis = "...";
  let lo = 0;
  let hi = src.length;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    const candidate = `${src.slice(0, mid)}${ellipsis}`;
    if (textMeasureCtx.measureText(candidate).width <= width) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return {
    text: `${src.slice(0, lo)}${ellipsis}`,
    truncated: true,
    full: src,
  };
}

function renderAssistant(frame, rawText, thinkMode, options = {}) {
  const streaming = options.streaming === true;
  if (!thinkMode) {
    const plain = String(rawText || "");
    if (streaming) {
      setAssistantBody(frame, plain, { typesetMath: false });
    } else {
      setAssistantBody(frame, plain);
    }
    frame.thinkingLive.classList.add("hidden");
    frame.thinkingLive.classList.remove("expandable");
    frame.thinkingLive.classList.remove("live-expanded");
    frame.thinkingPreview.textContent = "";
    frame.thinkingPreview.classList.remove("hidden");
    frame.thinkingPreviewFull.classList.add("hidden");
    frame.thinkingPreviewFull.textContent = "";
    frame.thinkingPreviewHint.classList.add("hidden");
    frame.liveCanExpand = false;
    frame.liveExpanded = false;
    frame.liveThoughtText = "";
    frame.thought.classList.add("hidden");
    frame.expanded = false;
    frame.thoughtToggle.textContent = "Show trace";
    frame.thoughtTitle.textContent = "Thinking Trace (Internal)";
    frame.thoughtRecent.classList.add("hidden");
    frame.thoughtRecent.textContent = "";
    frame.thoughtFull.classList.add("hidden");
    frame.thoughtFull.textContent = "";
    return;
  }

  const parsed = extractThought(rawText, thinkMode);
  const hasAnswer = Boolean(parsed.answer && parsed.answer.trim().length > 0);
  const thoughtText = normalizeThoughtText(parsed.thought);
  const livePreviewInfo = truncateToSingleLineByWidth(
    currentThoughtSegment(thoughtText),
    frame.thinkingPreview,
    frame.previewMetrics,
  );
  const livePreview = livePreviewInfo.text;

  if (thinkMode && !hasAnswer) {
    frame.thinkingLive.classList.remove("hidden");
    frame.thinkingPreview.textContent = livePreview || "Awaiting reasoning tokens...";
    frame.liveThoughtText = thoughtText;
    frame.liveCanExpand = Boolean(thoughtText && (livePreviewInfo.truncated || frame.liveExpanded));
    if (!thoughtText) {
      frame.liveExpanded = false;
    }
    frame.thinkingLive.classList.toggle("expandable", frame.liveCanExpand);
    if (frame.liveExpanded) {
      frame.thinkingLive.classList.add("live-expanded");
      frame.thinkingPreview.classList.add("hidden");
      frame.thinkingPreviewFull.textContent = thoughtText;
      frame.thinkingPreviewFull.classList.remove("hidden");
      frame.thinkingPreviewHint.textContent = "Click to collapse";
      frame.thinkingPreviewHint.classList.remove("hidden");
    } else {
      frame.thinkingLive.classList.remove("live-expanded");
      frame.thinkingPreview.classList.remove("hidden");
      frame.thinkingPreviewFull.classList.add("hidden");
      frame.thinkingPreviewFull.textContent = "";
      if (frame.liveCanExpand) {
        frame.thinkingPreviewHint.textContent = "Click to expand";
        frame.thinkingPreviewHint.classList.remove("hidden");
      } else {
        frame.thinkingPreviewHint.classList.add("hidden");
      }
    }
    frame.thought.classList.add("hidden");
    frame.expanded = false;
    frame.thoughtToggle.textContent = "Show trace";
    frame.thoughtTitle.textContent = "Thinking Output (Initial)";
  } else {
    frame.thinkingLive.classList.add("hidden");
    frame.thinkingLive.classList.remove("expandable");
    frame.thinkingLive.classList.remove("live-expanded");
    frame.thinkingPreview.textContent = "";
    frame.thinkingPreview.classList.remove("hidden");
    frame.thinkingPreviewFull.classList.add("hidden");
    frame.thinkingPreviewFull.textContent = "";
    frame.thinkingPreviewHint.classList.add("hidden");
    frame.liveCanExpand = false;
    frame.liveExpanded = false;
    frame.liveThoughtText = "";
  }

  if (hasAnswer) {
    if (streaming) {
      setAssistantBody(frame, parsed.answer, { typesetMath: false });
    } else {
      setAssistantBody(frame, parsed.answer);
    }
  } else if (!thinkMode) {
    const fallback = parsed.answer || String(rawText || "");
    if (streaming) {
      setAssistantBody(frame, fallback, { typesetMath: false });
    } else {
      setAssistantBody(frame, fallback);
    }
  } else {
    frame.body.textContent = "";
  }

  if (thinkMode && hasAnswer && thoughtText) {
    frame.thought.classList.remove("hidden");
    frame.thoughtRecent.classList.add("hidden");
    frame.thoughtRecent.textContent = "";
    frame.thoughtFull.textContent = thoughtText;
    frame.thoughtTitle.textContent = "Thinking Trace (Internal)";
    if (frame.expanded) {
      frame.thoughtFull.classList.remove("hidden");
    } else {
      frame.thoughtFull.classList.add("hidden");
    }
  } else {
    frame.thought.classList.add("hidden");
    frame.expanded = false;
    frame.thoughtToggle.textContent = "Show trace";
    frame.thoughtTitle.textContent = "Thinking Trace (Internal)";
    frame.thoughtRecent.classList.add("hidden");
    frame.thoughtRecent.textContent = "";
    frame.thoughtFull.classList.add("hidden");
    frame.thoughtFull.textContent = "";
  }
}

function createStreamRenderScheduler(frame, thinkMode) {
  let latestText = "";
  let pending = false;
  let lastRenderTs = 0;

  const flush = (force = false) => {
    pending = false;
    const now = performance.now();
    if (!force && now - lastRenderTs < STREAM_RENDER_INTERVAL_MS) {
      schedule(false);
      return;
    }
    lastRenderTs = now;
    renderAssistant(frame, latestText, thinkMode, { streaming: true });
    scrollMessages();
  };

  const schedule = (force = false) => {
    if (force) {
      flush(true);
      return;
    }
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => flush(false));
  };

  return {
    push(text) {
      latestText = String(text || "");
      schedule(false);
    },
    flush() {
      flush(true);
    },
  };
}

function updateSliderReadouts() {
  if (els.temperatureValue) {
    els.temperatureValue.textContent = Number(els.temperature.value).toFixed(2);
  }
  if (els.topPValue) {
    els.topPValue.textContent = Number(els.topP.value).toFixed(2);
  }
  if (els.repetitionPenaltyValue && els.repetitionPenalty) {
    els.repetitionPenaltyValue.textContent = Number(els.repetitionPenalty.value).toFixed(2);
  }
  if (els.maxNewValue) {
    els.maxNewValue.textContent = String(Math.round(Number(els.maxNew.value)));
  }
}

function historyPayload() {
  const history = conversationHistory.map((m) => ({ role: m.role, content: m.content }));
  if (!history.length) return history;

  if (history.length <= HISTORY_RECENT_MESSAGES) {
    return history;
  }

  const older = history.slice(0, history.length - HISTORY_RECENT_MESSAGES);
  const recent = history.slice(-HISTORY_RECENT_MESSAGES);

  const start = Math.max(0, older.length - HISTORY_SUMMARY_MAX_ITEMS);
  const summaryItems = [];
  for (let i = start; i < older.length; i += 1) {
    const turn = older[i];
    const role = turn.role === "user" ? "User" : "Assistant";
    const normalized = String(turn.content || "").replace(/\s+/g, " ").trim();
    if (!normalized) continue;
    const clipped =
      normalized.length > 180 ? `${normalized.slice(0, 177).trimEnd()}...` : normalized;
    summaryItems.push(`- ${role}: ${clipped}`);
  }

  if (!summaryItems.length) return recent;

  let summaryText =
    "Conversation memory summary (older turns):\n" + summaryItems.join("\n");
  if (summaryText.length > HISTORY_SUMMARY_CHAR_CAP) {
    summaryText = `${summaryText.slice(0, HISTORY_SUMMARY_CHAR_CAP - 3).trimEnd()}...`;
  }

  return [{ role: "system", content: summaryText }, ...recent];
}

function wrapThoughtAndAnswer(thought, answer) {
  const t = String(thought || "").trim();
  const a = String(answer || "").trim();
  if (t && a) {
    return `<thinking>\n${t}\n</thinking>\n\n${a}`;
  }
  if (a) return a;
  if (t) return `<thinking>\n${t}\n</thinking>`;
  return "";
}

async function typeInto(node, text) {
  const words = text.split(/(\s+)/);
  node.textContent = "";
  for (let i = 0; i < words.length; i += 1) {
    node.textContent += words[i];
    if (i % 3 === 0) {
      await new Promise((r) => setTimeout(r, 8));
    }
  }
}

function setBusy(nextBusy) {
  if (nextBusy && !busy) {
    loadStdTarget = LOAD_BUSY_STD;
  }
  if (!nextBusy && busy) {
    loadStdTarget = LOAD_IDLE_STD;
  }
  busy = nextBusy;
  els.sendBtn.disabled = busy;
  els.signalState.textContent = busy ? "running" : "idle";
}

function randomNormal() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function nextAr3LoadSample() {
  const transitionRate = loadStdTarget > loadStdCurrent ? LOAD_TRANSITION_UP : LOAD_TRANSITION_DOWN;
  loadStdCurrent += (loadStdTarget - loadStdCurrent) * transitionRate;

  const eps = randomNormal() * loadStdCurrent;
  const next =
    LOAD_AR3_COEFFS[0] * loadAr1 +
    LOAD_AR3_COEFFS[1] * loadAr2 +
    LOAD_AR3_COEFFS[2] * loadAr3 +
    eps;

  loadAr3 = loadAr2;
  loadAr2 = loadAr1;
  loadAr1 = Math.max(-2.4, Math.min(2.4, next));
  return loadAr1;
}

function updateMetrics(metrics) {
  if (!metrics) return;
  els.metrics.tps.textContent = (metrics.tokens_per_sec ?? 0).toFixed(2);
  els.metrics.ftl.textContent = `${Math.round(metrics.first_token_ms ?? 0)} ms`;
  els.metrics.out.textContent = `${Math.round(metrics.generated_tokens ?? 0)}`;
  els.metrics.in.textContent = `${Math.round(metrics.prompt_tokens ?? 0)}`;
  const mem = Number(metrics.model_memory_mb ?? 0);
  els.metrics.mem.textContent = `${mem.toFixed(1)} MB`;
  els.metrics.ratio.textContent = `${Number(metrics.compression_ratio ?? 0).toFixed(2)}x`;
  if (metrics.precision) {
    els.precisionTag.textContent = metrics.precision;
  }
}

function updateDetails(meta, config) {
  els.details.name.textContent = `Model: ${meta?.model?.name ?? "-"}`;
  els.details.arch.textContent = `Arch: ${config?.model_type ?? "qwen3"} · ${config?.num_hidden_layers ?? "?"} layers · ${config?.hidden_size ?? "?"} hidden · CPU WASM`;
  els.details.quant.textContent = `Quant: ${meta?.runtime?.quant_method ?? "gptq"} / ${meta?.runtime?.weight_precision ?? "int4"} + ${meta?.runtime?.act_precision ?? "fp16"}`;
  els.details.path.textContent = `Path: ${meta?.model?.path ?? "-"}`;
  if (els.details.data) {
    const trainingData =
      meta?.model?.training_data ||
      "Trained on 1.2 million PubMed publications related to cognitive neuroscience";
    els.details.data.textContent = `Data: ${trainingData}`;
  }

  const method = meta?.runtime?.quant_method?.toUpperCase?.() || "GPTQ";
  const bits = meta?.runtime?.bits ?? 4;
  const actPrecision = meta?.runtime?.act_precision?.toUpperCase?.() || "FP16";
  els.precisionTag.textContent = `${method} W${bits}A${actPrecision.replace("FP", "")}`;
}

function drawSignal(ts = performance.now()) {
  const canvas = els.signalCanvas;
  if (!canvas) return;
  if (ts - loadLastDrawTs < SIGNAL_DRAW_INTERVAL_MS) {
    requestAnimationFrame(drawSignal);
    return;
  }
  loadLastDrawTs = ts;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);

  const grad = ctx.createLinearGradient(0, 0, width, 0);
  grad.addColorStop(0, "rgba(176, 176, 176, 0.72)");
  grad.addColorStop(1, "rgba(52, 211, 153, 0.86)");

  ctx.strokeStyle = "rgba(220, 220, 220, 0.16)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();

  ctx.lineWidth = 2;
  ctx.strokeStyle = grad;
  ctx.beginPath();

  if (loadTrace.length !== width) {
    loadTrace = new Array(width).fill(0);
  }

  const now = performance.now();
  if (loadLastTs === 0) {
    loadLastTs = now;
  }
  const dtMs = Math.min(120, Math.max(0, now - loadLastTs));
  loadLastTs = now;
  loadAccumulatorMs += dtMs;

  while (loadAccumulatorMs >= LOAD_SAMPLE_INTERVAL_MS) {
    loadAccumulatorMs -= LOAD_SAMPLE_INTERVAL_MS;
    const sample = nextAr3LoadSample();
    loadTrace.push(sample);
    if (loadTrace.length > width) {
      loadTrace.shift();
    }
  }

  const yRange = height * 0.48;
  for (let x = 0; x < width; x += 1) {
    const sample = loadTrace[x] ?? 0;
    const y = height / 2 + sample * yRange * LOAD_Y_SCALE;
    if (x === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }

  ctx.stroke();
  requestAnimationFrame(drawSignal);
}

async function onSubmit(event) {
  event.preventDefault();
  if (busy) return;
  if (!runtimeReady) {
    addMessage("assistant", "WASM runtime is not ready yet. Resolve startup error and reload.");
    return;
  }

  const prompt = els.prompt.value.trim();
  if (!prompt) return;
  if (els.controlsDrawer && isMobileLayout()) {
    els.controlsDrawer.open = false;
  }

  setBusy(true);
  activeAbortController = new AbortController();
  addMessage("user", prompt);
  els.prompt.value = "";
  setStatus("Running local WASM inference");

  const thinkMode = Boolean(els.thinkMode?.checked);
  const useHistory = Boolean(els.historyMode?.checked);
  const repetitionPenalty = Number(els.repetitionPenalty?.value ?? 1.0);
  const maxNewTokens = Math.round(Number(els.maxNew.value));
  const streamMaxTokens = thinkMode ? Math.min(maxNewTokens, THINK_TOKEN_CAP) : maxNewTokens;
  if (!useHistory) {
    conversationHistory.length = 0;
  }
  const priorHistory = useHistory ? historyPayload() : [];
  const assistantFrame = createAssistantFrame();
  if (thinkMode) {
    assistantFrame.thinkingLive.classList.remove("hidden");
    initThinkingPreviewSizing(assistantFrame);
    assistantFrame.thinkingPreview.textContent = "Awaiting reasoning tokens...";
    assistantFrame.body.textContent = "";
    assistantFrame.thought.classList.add("hidden");
    assistantFrame.thoughtToggle.textContent = "Show trace";
    assistantFrame.expanded = false;
    assistantFrame.thoughtRecent.textContent = "";
    assistantFrame.thoughtFull.textContent = "";
  }
  let streamed = false;
  let streamedRaw = "";
  let finalResult = null;
  let finalRawText = "";
  let thinkTimedOut = false;
  let thinkTimeoutId = null;
  let lastThinkAnswerScanTs = 0;
  let answerStreamingStarted = false;
  const streamRenderer = createStreamRenderScheduler(assistantFrame, thinkMode);

  try {
    if (thinkMode) {
      thinkTimeoutId = setTimeout(() => {
        if (answerStreamingStarted) return;
        const latest = extractThought(streamedRaw, true);
        if (latest.answer && latest.answer.trim().length > 0) {
          answerStreamingStarted = true;
          return;
        }
        thinkTimedOut = true;
        if (activeAbortController) {
          activeAbortController.abort();
        }
      }, THINK_TIME_LIMIT_MS);
    }

    const result = await engine.generateStream(
      prompt,
      {
        temperature: Number(els.temperature.value),
        maxNewTokens: streamMaxTokens,
        topP: Number(els.topP.value),
        repetitionPenalty,
        topK: 20,
        thinkMode,
        history: priorHistory,
        signal: activeAbortController.signal,
      },
      (delta, fullText) => {
        if (!delta) return;
        if (!streamed) {
          streamed = true;
        }
        if (fullText) {
          streamedRaw = String(fullText);
        } else {
          streamedRaw += String(delta);
        }
        if (thinkMode && !answerStreamingStarted) {
          const now = performance.now();
          const shouldScan =
            now - lastThinkAnswerScanTs >= THINK_ANSWER_SCAN_INTERVAL_MS ||
            String(delta).includes("</think>") ||
            String(delta).includes("</thinking>");
          if (shouldScan) {
            lastThinkAnswerScanTs = now;
            const partial = extractThought(streamedRaw, true);
            if (partial.answer && partial.answer.trim().length > 0) {
              answerStreamingStarted = true;
              if (thinkTimeoutId) {
                clearTimeout(thinkTimeoutId);
                thinkTimeoutId = null;
              }
            }
          }
        }
        streamRenderer.push(streamedRaw);
      },
    );
    if (thinkTimeoutId) {
      clearTimeout(thinkTimeoutId);
      thinkTimeoutId = null;
    }
    finalResult = result;
    finalRawText = result.text || streamedRaw || "";
    const generatedTokens = Math.round(Number(finalResult?.metrics?.generated_tokens ?? 0));
    const thinkLimitReached = thinkMode && generatedTokens >= streamMaxTokens;

    streamRenderer.flush();

    if (!streamed && finalRawText) {
      renderAssistant(assistantFrame, finalRawText, thinkMode);
    } else if (result.text && streamedRaw !== result.text) {
      renderAssistant(assistantFrame, result.text, thinkMode);
      finalRawText = result.text;
    }

    if (thinkMode) {
      const parsedFinal = extractThought(finalRawText, true);
      if (!parsedFinal.answer || thinkLimitReached) {
        setStatus(
          thinkTimedOut
            ? "Thinking timed out, requesting final answer segment"
            : thinkLimitReached
              ? "Thinking cap reached, requesting final answer segment"
              : "Requesting final answer segment",
        );
        try {
          const recovered = await engine.generate(prompt, {
            temperature: Number(els.temperature.value),
            maxNewTokens,
            topP: Number(els.topP.value),
            repetitionPenalty,
            topK: 20,
            thinkMode: false,
            history: priorHistory,
          });
          const recoveredAnswer = String(recovered?.text || "").trim();
          if (recoveredAnswer) {
            const note = thinkTimedOut
              ? `[Note: thinking timed out at ${Math.round(THINK_TIME_LIMIT_MS / 1000)}s; final answer requested directly.]\n\n`
              : thinkLimitReached
                ? `[Note: internal thinking cap (${THINK_TOKEN_CAP} tokens) reached to keep latency bounded.]\n\n`
                : "";
            finalRawText = wrapThoughtAndAnswer(parsedFinal.thought, `${note}${recoveredAnswer}`);
            finalResult = {
              ...finalResult,
              mode: recovered.mode || finalResult?.mode,
              metrics: recovered.metrics || finalResult?.metrics,
            };
          }
        } catch (recoverErr) {
          console.error("Final-answer recovery failed:", recoverErr);
        }
      }
    }

    renderAssistant(assistantFrame, finalRawText, thinkMode);
    updateMetrics(finalResult?.metrics);
    setStatus(runtimeLabel(finalResult?.mode, "active"));

    if (useHistory) {
      const normalized = extractThought(finalRawText || "", thinkMode);
      const assistantContent = normalized.answer || String(finalRawText || "").trim();
      conversationHistory.push({ role: "user", content: prompt });
      if (assistantContent) {
        conversationHistory.push({ role: "assistant", content: assistantContent });
      }
      if (conversationHistory.length > MAX_HISTORY_MESSAGES) {
        conversationHistory.splice(0, conversationHistory.length - MAX_HISTORY_MESSAGES);
      }
    }
  } catch (err) {
    if (thinkTimeoutId) {
      clearTimeout(thinkTimeoutId);
      thinkTimeoutId = null;
    }
    console.error(err);
    streamRenderer.flush();
    if (err?.name === "AbortError" && thinkMode && thinkTimedOut) {
      setStatus("Thinking timed out, requesting final answer segment");
      try {
        const recovered = await engine.generate(prompt, {
          temperature: Number(els.temperature.value),
          maxNewTokens,
          topP: Number(els.topP.value),
          repetitionPenalty,
          topK: 20,
          thinkMode: false,
          history: priorHistory,
        });
        const recoveredAnswer = String(recovered?.text || "").trim();
        if (recoveredAnswer) {
          const timedOutParsed = extractThought(streamedRaw, true);
          finalRawText = wrapThoughtAndAnswer(
            timedOutParsed.thought,
            `[Note: thinking timed out at ${Math.round(THINK_TIME_LIMIT_MS / 1000)}s; final answer requested directly.]\n\n${recoveredAnswer}`,
          );
          renderAssistant(assistantFrame, finalRawText, true);
          updateMetrics(recovered.metrics);
          setStatus(runtimeLabel(recovered.mode, "active"));
          if (useHistory) {
            const normalized = extractThought(finalRawText || "", true);
            const assistantContent = normalized.answer || String(finalRawText || "").trim();
            conversationHistory.push({ role: "user", content: prompt });
            if (assistantContent) {
              conversationHistory.push({ role: "assistant", content: assistantContent });
            }
            if (conversationHistory.length > MAX_HISTORY_MESSAGES) {
              conversationHistory.splice(0, conversationHistory.length - MAX_HISTORY_MESSAGES);
            }
          }
        } else {
          assistantFrame.body.textContent = "Generation timed out and no final answer was produced.";
          assistantFrame.thinkingLive.classList.add("hidden");
          assistantFrame.thought.classList.add("hidden");
          setStatus("Error");
        }
      } catch (recoverErr) {
        assistantFrame.body.textContent = `Generation failed after thinking timeout: ${recoverErr.message || recoverErr}`;
        assistantFrame.thinkingLive.classList.add("hidden");
        assistantFrame.thought.classList.add("hidden");
        setStatus("Error");
      }
    } else {
      assistantFrame.thinkingLive.classList.add("hidden");
      if (err?.name === "AbortError") {
        assistantFrame.body.textContent = "Generation canceled.";
        setStatus("Canceled");
      } else {
        assistantFrame.body.textContent = `Generation failed: ${err.message}`;
        setStatus("Error");
      }
      assistantFrame.thought.classList.add("hidden");
    }
  } finally {
    activeAbortController = null;
    setBusy(false);
  }
}

async function boot() {
  drawSignal();
  updateSliderReadouts();
  syncControlsDrawerByViewport();
  setStatus("Downloading model and initializing WASM CPU");
  els.sendBtn.disabled = true;
  setStartupVisible(true);
  updateStartupView({
    title: "Preparing NeuroLM (WASM CPU)",
    stage: "Downloading model weights and loading runtime...",
    sub: "First load may take 1-3 minutes. Later loads should come from cache.",
    progress: null,
  });

  if (navigator.storage?.persist) {
    navigator.storage.persist().catch(() => {});
  }

  if (els.startupRetry) {
    els.startupRetry.addEventListener("click", () => window.location.reload());
  }
  if (els.controlsDrawer) {
    const mq = window.matchMedia(MOBILE_LAYOUT_QUERY);
    const onViewportChange = () => syncControlsDrawerByViewport();
    if (typeof mq.addEventListener === "function") {
      mq.addEventListener("change", onViewportChange);
    } else if (typeof mq.addListener === "function") {
      mq.addListener(onViewportChange);
    }
  }

  if (typeof engine.setInitProgressHandler === "function") {
    engine.setInitProgressHandler((payload) => {
      const stage = String(payload?.stage || "");
      const modelFile = String(payload?.modelFile || "");
      const source = payload?.source === "cache" ? "cache" : "network";
      const loadedBytes = Number(payload?.loadedBytes || 0);
      const totalBytes = Number(payload?.totalBytes || 0);
      const explicitPct = Number(payload?.progress || NaN);
      const computedPct =
        Number.isFinite(explicitPct)
          ? explicitPct
          : totalBytes > 0
            ? (loadedBytes / totalBytes) * 100
            : NaN;

      if (stage === "downloading") {
        const progressText =
          totalBytes > 0
            ? `${formatMb(loadedBytes * 1)} / ${formatMb(totalBytes)} MB`
            : `${formatMb(loadedBytes)} MB downloaded`;
        updateStartupView({
          stage: `Downloading ${modelFile || "model"} (${progressText})`,
          sub: "Please keep this tab open while the model downloads.",
          progress: computedPct,
        });
        return;
      }

      if (stage === "cache-hit") {
        updateStartupView({
          stage: `Loading ${modelFile || "model"} from local browser cache`,
          sub: "Model cache hit detected. Startup should be faster.",
          progress: 100,
        });
        return;
      }

      if (stage === "model-loaded") {
        updateStartupView({
          stage: `Model loaded (${modelFile || "GGUF"}) from ${source}`,
          sub: "Initializing inference runtime...",
          progress: 100,
        });
        return;
      }

      if (stage === "init") {
        updateStartupView({
          stage: "Preparing WASM runtime",
          sub: "WASM CPU runtime is starting...",
          progress: null,
        });
      }
    });
  }

  try {
    const init = await engine.init();
    runtimeReady = true;
    updateDetails(init.meta, init.config);
    updateMetrics({
      model_memory_mb: init.meta?.model?.safetensors_file_mb ?? 0,
      compression_ratio: init.meta?.model?.compression_ratio_vs_bf16 ?? 0,
      precision: `${(init.meta?.runtime?.quant_method || "gptq").toUpperCase()} W${init.meta?.runtime?.bits || 4}A16`,
    });

    setStatus(runtimeLabel(init.mode, "ready"));
    if (init.mode === "unavailable") {
      const detail = "WASM runtime unavailable.";
      addMessage(
        "assistant",
        `${detail} Check that model files exist under ./model and wasm assets under ./pkg.`,
      );
    }
    if (!busy) {
      els.sendBtn.disabled = false;
    }
    setStartupVisible(false);
  } catch (err) {
    console.error(err);
    runtimeReady = false;
    setStatus("Initialization failed");
    const msg = String(err?.message || err || "Unknown startup error");
    updateStartupView({
      stage: "Initialization failed",
      sub: msg,
      progress: null,
      failed: true,
    });
    setStartupVisible(true);
    addMessage(
      "assistant",
      `Startup error: ${msg}\n\nExpected model files in /chat/model or /model with config.json, generation_config.json, tokenizer.json and a GGUF file (model-q4_k_m.gguf, model-q4_0.gguf, or model.gguf).`,
    );
  }

  els.form.addEventListener("submit", onSubmit);
  [els.temperature, els.topP, els.repetitionPenalty, els.maxNew].forEach((input) => {
    if (!input) return;
    input.addEventListener("input", updateSliderReadouts);
  });
  if (els.historyMode) {
    els.historyMode.addEventListener("change", () => {
      if (!els.historyMode.checked) {
        conversationHistory.length = 0;
      }
    });
  }
  els.prompt.addEventListener("keydown", (event) => {
    const isEnter = event.key === "Enter" || event.code === "NumpadEnter";
    const plainEnter =
      isEnter &&
      !event.shiftKey &&
      !event.ctrlKey &&
      !event.metaKey &&
      !event.altKey &&
      !event.isComposing;
    if (!plainEnter) return;
    event.preventDefault();
    if (!busy) {
      els.form.requestSubmit();
    }
  });

  window.addEventListener("keydown", (event) => {
    if (!busy || !activeAbortController) return;
    const isCtrlC = (event.ctrlKey || event.metaKey) && String(event.key).toLowerCase() === "c";
    if (!isCtrlC) return;
    event.preventDefault();
    activeAbortController.abort();
  });
}

boot();
