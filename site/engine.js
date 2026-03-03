export class LocalNeuroEngine {
  constructor(options = {}) {
    const buildTag = "20260302-09";
    this.modelBaseUrl = options.modelBaseUrl || "./model";
    this.metaUrl = options.metaUrl || null;
    this.workerUrl =
      options.workerUrl || new URL(`./wasm-worker.js?v=${buildTag}`, import.meta.url);
    // Default baseline for "original model size" used in compression metric:
    // 1.5 GiB ~= 1536 MiB, matching the expected source checkpoint size.
    this.originalModelBytes = Number(options.originalModelBytes ?? 1.5 * 1024 * 1024 * 1024);

    this.mode = "initializing";
    this.meta = null;
    this.config = null;
    this.generationConfig = null;
    this.tokenizerModel = null;

    this.worker = null;
    this.workerReady = false;
    this.lastInitError = null;
    this.reqId = 0;
    this.pending = new Map();
    this.streamPending = new Map();
    this.modelInitPromise = null;
    this.initProgressHandler = null;

    // History compression controls.
    this.historyStrategy = String(options.historyStrategy ?? "questions_only");
    this.historyTokenBudget = Math.max(128, Number(options.historyTokenBudget ?? 320));
    this.historyRecentTurns = Math.max(1, Number(options.historyRecentTurns ?? 4));
    this.historyRecentCharCap = Math.max(80, Number(options.historyRecentCharCap ?? 220));
    this.historyOlderCharCap = Math.max(40, Number(options.historyOlderCharCap ?? 140));
    this.historySummaryTokenBudget = Math.max(16, Number(options.historySummaryTokenBudget ?? 80));
    this.historyUseSummary = options.historyUseSummary === true;
  }

  setInitProgressHandler(handler) {
    this.initProgressHandler = typeof handler === "function" ? handler : null;
  }

  async init() {
    const { baseUrl, config } = await this.#resolveModelBaseUrl();
    this.modelBaseUrl = baseUrl;
    this.config = config;
    const [generation, tokenizerModel, meta] = await Promise.all([
      this.#fetchJson(`${this.modelBaseUrl}/generation_config.json`),
      this.#fetchJson(`${this.modelBaseUrl}/tokenizer.json`),
      this.#loadMetaFallback(),
    ]);

    this.generationConfig = generation;
    this.tokenizerModel = tokenizerModel;
    this.meta = meta;

    await this.#ensureWasmReady();

    return {
      mode: this.mode,
      meta: this.meta,
      config: this.config,
      generation: this.generationConfig,
      tokenizerConfig: this.tokenizerModel,
    };
  }

  async generate(prompt, options = {}) {
    const maxNewTokens = Number(options.maxNewTokens ?? 8192);
    const temperature = Number(options.temperature ?? 0.10);
    const topP = Number(options.topP ?? 0.95);
    const repetitionPenalty = Number(
      options.repetitionPenalty ?? this.generationConfig?.repetition_penalty ?? 1.10,
    );
    const topK = Number(options.topK ?? this.generationConfig?.top_k ?? 20);
    const thinkMode = options.thinkMode === true;
    const history = this.#sanitizeHistory(options.history);
    const signal = options.signal;

    if (signal?.aborted) {
      throw new DOMException("Generation aborted", "AbortError");
    }

    const promptWithHistory = this.#composePromptWithHistory(prompt, history);
    const promptWithDirective = this.#applyThinkingDirective(promptWithHistory, thinkMode);
    const tuned = this.#maybeTuneSampling(temperature, topP, topK, repetitionPenalty, thinkMode);
    return this.#generateViaWasm(
      promptWithDirective,
      maxNewTokens,
      tuned.temperature,
      tuned.topP,
      tuned.topK,
      tuned.repetitionPenalty,
      thinkMode,
      signal,
    );
  }

  async generateStream(prompt, options = {}, onToken = null) {
    const maxNewTokens = Number(options.maxNewTokens ?? 8192);
    const temperature = Number(options.temperature ?? 0.10);
    const topP = Number(options.topP ?? 0.95);
    const repetitionPenalty = Number(
      options.repetitionPenalty ?? this.generationConfig?.repetition_penalty ?? 1.10,
    );
    const topK = Number(options.topK ?? this.generationConfig?.top_k ?? 20);
    const thinkMode = options.thinkMode === true;
    const history = this.#sanitizeHistory(options.history);
    const signal = options.signal;

    if (signal?.aborted) {
      throw new DOMException("Generation aborted", "AbortError");
    }

    const promptWithHistory = this.#composePromptWithHistory(prompt, history);
    const promptWithDirective = this.#applyThinkingDirective(promptWithHistory, thinkMode);
    const tuned = this.#maybeTuneSampling(temperature, topP, topK, repetitionPenalty, thinkMode);
    const wasmPayload = await this.#generateViaWasmStream(
      promptWithDirective,
      maxNewTokens,
      tuned.temperature,
      tuned.topP,
      tuned.topK,
      tuned.repetitionPenalty,
      thinkMode,
      signal,
      onToken,
    );
    wasmPayload.mode = wasmPayload.mode || "candle-wasm";
    this.mode = wasmPayload.mode;
    return wasmPayload;
  }

  async #loadMetaFallback() {
    const metaCandidates = [];
    if (this.metaUrl) {
      metaCandidates.push(this.metaUrl);
    }
    metaCandidates.push(`${this.modelBaseUrl}/meta.json`);

    for (const candidate of metaCandidates) {
      try {
        const meta = await this.#fetchJson(candidate);
        return this.#normalizeMeta(meta);
      } catch {
        // continue candidate probing
      }
    }

    {
      let modelBytes = null;
      const candidates = ["model-q4_k_m.gguf", "model-q4_0.gguf", "model.gguf", "model.safetensors"];
      for (const name of candidates) {
        try {
          const head = await fetch(`${this.modelBaseUrl}/${name}`, {
            method: "HEAD",
            cache: "default",
          });
          if (head.ok) {
            const raw = head.headers.get("content-length");
            if (raw) {
              const parsed = Number(raw);
              if (Number.isFinite(parsed) && parsed > 0) {
                modelBytes = parsed;
                break;
              }
            }
          }
        } catch {
          // best-effort metadata probe only
        }
      }
      return this.#normalizeMeta(this.#deriveMeta(modelBytes));
    }
  }

  async #resolveModelBaseUrl() {
    const candidates = [];
    const push = (v) => {
      const value = String(v || "").trim();
      if (!value) return;
      if (!candidates.includes(value)) {
        candidates.push(value);
      }
    };

    push(this.modelBaseUrl);
    push("./model");
    push("../model");

    let lastErr = null;
    for (const baseUrl of candidates) {
      try {
        const config = await this.#fetchJson(`${baseUrl}/config.json`);
        return { baseUrl, config };
      } catch (err) {
        lastErr = err;
      }
    }

    throw new Error(
      `Model files not found. Tried: ${candidates.join(", ")}. Last error: ${lastErr?.message || "unknown"}`,
    );
  }

  #deriveMeta(modelBytes) {
    const quant = this.config?.quantization_config || {};
    const bits = Number(quant.bits ?? 4);
    const groupSize = Number(quant.group_size ?? 128);
    const quantMethod = "gguf";
    const modelMb = Number.isFinite(modelBytes) && modelBytes > 0 ? modelBytes / (1024 * 1024) : 0;
    const originalBytes = Number.isFinite(this.originalModelBytes) && this.originalModelBytes > 0
      ? this.originalModelBytes
      : 0;
    const compressionRatio =
      Number.isFinite(modelBytes) && modelBytes > 0 && originalBytes > 0
        ? originalBytes / modelBytes
        : 0;
    const originalModelMb = originalBytes > 0 ? originalBytes / (1024 * 1024) : 0;

    return {
      model: {
        name: "Qwen3-0.6B-pubmed-neuroimaging-dapt-gptq-w4a16",
        path: `${this.modelBaseUrl}/model-q4_k_m.gguf`,
        safetensors_file_mb: modelMb,
        original_model_mb: originalModelMb,
        compression_ratio_vs_bf16: compressionRatio,
      },
      runtime: {
        quant_method: quantMethod,
        bits,
        group_size: groupSize,
        weight_precision: `int${bits}`,
        act_precision: "fp16",
      },
    };
  }

  #normalizeMeta(meta) {
    const m = meta && typeof meta === "object" ? meta : {};
    const model = m.model && typeof m.model === "object" ? m.model : {};
    const runtime = m.runtime && typeof m.runtime === "object" ? m.runtime : {};

    const bits = Number(runtime.bits ?? 4);
    const quantMethod = String(runtime.quant_method || "gptq");

    const modelMb = Number(model.safetensors_file_mb ?? 0);
    const originalModelMb = Number(model.original_model_mb ?? (this.originalModelBytes / (1024 * 1024)));
    let compressionRatio = Number(model.compression_ratio_vs_bf16 ?? 0);
    if ((!Number.isFinite(compressionRatio) || compressionRatio <= 0) && modelMb > 0 && originalModelMb > 0) {
      compressionRatio = originalModelMb / modelMb;
    }

    return {
      ...m,
      model: {
        name: String(model.name || "Qwen3-0.6B-pubmed-neuroimaging-dapt-gptq-w4a16"),
        path: String(model.path || `${this.modelBaseUrl}/model-q4_k_m.gguf`),
        safetensors_file_mb: modelMb,
        original_model_mb: Number.isFinite(originalModelMb) ? originalModelMb : 0,
        compression_ratio_vs_bf16: Number.isFinite(compressionRatio) ? compressionRatio : 0,
      },
      runtime: {
        quant_method: quantMethod,
        bits,
        group_size: Number(runtime.group_size ?? 128),
        weight_precision: String(runtime.weight_precision || `int${bits}`),
        act_precision: String(runtime.act_precision || "fp16"),
      },
    };
  }

  async #initWasmRuntime() {
    await this.#ensureWorker();
    try {
      await this.#rpc("init", {
        config: this.config,
        generation: this.generationConfig,
        tokenizer: this.tokenizerModel,
        meta: this.meta,
        modelBaseUrl: this.modelBaseUrl,
      });
      this.workerReady = true;
      this.mode = "candle-wasm";
      this.lastInitError = null;
    } catch (err) {
      this.workerReady = false;
      this.mode = "unavailable";
      this.lastInitError = err?.message || String(err);
      throw err;
    }
  }

  async #ensureWasmReady() {
    if (this.workerReady) return;
    if (!this.config || !this.generationConfig || !this.tokenizerModel || !this.meta) {
      throw new Error("WASM runtime configuration is incomplete");
    }
    if (!this.modelInitPromise) {
      this.modelInitPromise = this.#initWasmRuntime().finally(() => {
        this.modelInitPromise = null;
      });
    }
    await this.modelInitPromise;
  }

  #abortActiveWorker(reason = "Generation aborted") {
    if (this.worker) {
      try {
        this.worker.terminate();
      } catch {
        // no-op
      }
      this.worker = null;
    }
    this.workerReady = false;
    this.mode = "initializing";
    this.lastInitError = reason;
    this.pending.clear();
  }

  async #generateViaWasm(
    prompt,
    maxNewTokens,
    temperature,
    topP,
    topK,
    repetitionPenalty,
    enableThinking,
    signal,
  ) {
    if (signal?.aborted) {
      throw new DOMException("Generation aborted", "AbortError");
    }
    await this.#ensureWasmReady();
    const parsed = await this.#rpc("generate", {
      prompt,
      maxNewTokens,
      temperature,
      topP,
      topK,
      repetitionPenalty,
      enableThinking: enableThinking === true,
    });
    parsed.mode = parsed.mode || "candle-wasm";
    this.mode = parsed.mode;
    return parsed;
  }

  async #generateViaWasmStream(
    prompt,
    maxNewTokens,
    temperature,
    topP,
    topK,
    repetitionPenalty,
    enableThinking,
    signal,
    onToken,
  ) {
    await this.#ensureWasmReady();
    await this.#ensureWorker();
    return new Promise((resolve, reject) => {
      const id = ++this.reqId;
      let settled = false;

      const cleanup = () => {
        const pending = this.streamPending.get(id);
        if (pending?.signal && pending?.abortHandler) {
          pending.signal.removeEventListener("abort", pending.abortHandler);
        }
        this.streamPending.delete(id);
      };

      const resolveOnce = (result) => {
        if (settled) return;
        settled = true;
        cleanup();
        resolve(result);
      };

      const rejectOnce = (err) => {
        if (settled) return;
        settled = true;
        cleanup();
        reject(err);
      };

      const abortHandler = () => {
        this.#abortActiveWorker("Generation aborted by user");
        rejectOnce(new DOMException("Generation aborted", "AbortError"));
      };

      if (signal) {
        if (signal.aborted) {
          abortHandler();
          return;
        }
        signal.addEventListener("abort", abortHandler, { once: true });
      }

      this.streamPending.set(id, {
        resolve: resolveOnce,
        reject: rejectOnce,
        onToken,
        signal,
        abortHandler,
      });

      this.worker.postMessage({
        id,
        type: "generate_stream",
        payload: {
          prompt,
          maxNewTokens,
          temperature,
          topP,
          topK,
          repetitionPenalty,
          enableThinking: enableThinking === true,
        },
      });
    });
  }

  async #fetchJson(url) {
    const resp = await fetch(url, { cache: "default" });
    if (!resp.ok) {
      throw new Error(`Request failed (${resp.status}) for ${url}`);
    }
    return resp.json();
  }

  async #ensureWorker() {
    if (this.worker) return;
    this.worker = new Worker(this.workerUrl, { type: "module" });

    this.worker.onmessage = (event) => {
      const data = event.data || {};
      if (data.event === "init-progress") {
        if (typeof this.initProgressHandler === "function") {
          try {
            this.initProgressHandler(data.payload || {});
          } catch {
            // no-op
          }
        }
        return;
      }
      const { id, ok, result, error, stream, event: streamEvent, delta, text } = data;

      if (stream === true) {
        const pendingStream = this.streamPending.get(id);
        if (!pendingStream) return;

        if (streamEvent === "token") {
          if (typeof pendingStream.onToken === "function") {
            pendingStream.onToken(String(delta || ""), typeof text === "string" ? text : "");
          }
          return;
        }

        if (streamEvent === "done") {
          pendingStream.resolve(result);
          return;
        }

        if (streamEvent === "error") {
          pendingStream.reject(new Error(error || "WASM stream failed"));
        }
        return;
      }

      const pending = this.pending.get(id);
      if (!pending) return;
      this.pending.delete(id);
      if (ok) pending.resolve(result);
      else pending.reject(new Error(error || "WASM worker request failed"));
    };

    this.worker.onerror = (event) => {
      const msg = event?.message || "WASM worker crashed";
      for (const pending of this.pending.values()) {
        pending.reject(new Error(msg));
      }
      this.pending.clear();
      for (const pending of this.streamPending.values()) {
        pending.reject(new Error(msg));
      }
      this.streamPending.clear();
      this.workerReady = false;
      this.mode = "candle-wasm-error";
      this.lastInitError = msg;
    };
  }

  async #rpc(type, payload) {
    await this.#ensureWorker();
    return new Promise((resolve, reject) => {
      const id = ++this.reqId;
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, type, payload });
    });
  }

  #sanitizeHistory(history) {
    if (!Array.isArray(history)) return [];
    const out = [];
    for (const item of history) {
      if (!item || typeof item !== "object") continue;
      const role = String(item.role || "").trim().toLowerCase();
      if (role !== "user" && role !== "assistant" && role !== "system") continue;
      const content = String(item.content || "").trim();
      if (!content) continue;
      out.push({ role, content });
    }
    const maxMessages = 24;
    if (out.length > maxMessages) {
      return out.slice(out.length - maxMessages);
    }
    return out;
  }

  #normalizeText(text) {
    return String(text ?? "").replace(/\r/g, "").replace(/\s+/g, " ").trim();
  }

  #estimateTokens(text) {
    const clean = this.#normalizeText(text);
    if (!clean) return 0;
    // Common rough estimate used in prompt budgeting.
    return Math.max(1, Math.ceil(clean.length / 4));
  }

  #clipChars(text, maxChars) {
    const clean = this.#normalizeText(text);
    if (!clean) return "";
    const cap = Math.max(16, Number(maxChars || 0));
    if (clean.length <= cap) return clean;
    return `${clean.slice(0, Math.max(1, cap - 3)).trimEnd()}...`;
  }

  #truncateToTokenBudget(text, tokenBudget) {
    const clean = this.#normalizeText(text);
    const budget = Math.max(0, Number(tokenBudget || 0));
    if (!clean || budget <= 0) return "";
    const approxChars = Math.max(20, Math.floor(budget * 4));
    if (clean.length <= approxChars) return clean;
    return `${clean.slice(0, Math.max(1, approxChars - 3)).trimEnd()}...`;
  }

  #compressTurnContent(content, recent) {
    return this.#clipChars(
      content,
      recent ? this.historyRecentCharCap : this.historyOlderCharCap,
    );
  }

  #selectHistoryTurns(turns, prompt) {
    if (!Array.isArray(turns) || turns.length === 0) return [];
    void prompt;
    // Prioritize user questions. Keep system summary entries.
    if (this.historyStrategy === "questions_only") {
      return turns.filter((t) => t.role === "system" || t.role === "user");
    }
    return turns;
  }

  #buildDroppedSummary(droppedTurns, remainingBudget) {
    if (!Array.isArray(droppedTurns) || droppedTurns.length === 0) return "";
    const budget = Math.max(0, Math.floor(remainingBudget));
    if (budget < 16) return "";

    const snippets = [];
    const take = 6;
    const start = Math.max(0, droppedTurns.length - take);
    for (let i = start; i < droppedTurns.length; i += 1) {
      const turn = droppedTurns[i];
      const tag = turn.role === "user" ? "U" : "A";
      const snippet = this.#clipChars(turn.content, 90);
      if (!snippet) continue;
      snippets.push(`${tag}:${snippet}`);
    }
    if (!snippets.length) return "";

    const summaryRaw = `Earlier context (compressed): ${snippets.join(" | ")}`;
    const summaryBudget = Math.min(budget, this.historySummaryTokenBudget);
    return this.#truncateToTokenBudget(summaryRaw, summaryBudget);
  }

  #packHistory(turns, prompt) {
    const safePrompt = this.#normalizeText(prompt);
    const promptReserve = this.#estimateTokens(safePrompt) + 48;
    const totalBudget = this.historyTokenBudget;
    let remaining = Math.max(64, totalBudget - promptReserve);

    const selectedTurns = this.#selectHistoryTurns(turns, safePrompt);
    const selectedRev = [];
    const droppedRev = [];
    for (let age = 0, i = selectedTurns.length - 1; i >= 0; i -= 1, age += 1) {
      const turn = selectedTurns[i];
      const recent = age < this.historyRecentTurns;
      let content = this.#compressTurnContent(turn.content, recent);
      if (!content) continue;

      let lineCost = this.#estimateTokens(`${turn.role}:${content}`) + 2;
      if (lineCost > remaining) {
        // Preserve recent turns by truncating to fit the remaining budget.
        const fitBudget = Math.max(0, remaining - 4);
        content = this.#truncateToTokenBudget(content, fitBudget);
        if (!content) {
          break;
        }
        lineCost = this.#estimateTokens(`${turn.role}:${content}`) + 2;
      }

      if (lineCost > remaining) {
        break;
      }

      selectedRev.push({ role: turn.role, content });
      remaining -= lineCost;
      if (remaining <= 8) {
        break;
      }
    }

    const selected = selectedRev.reverse();
    const dropped = droppedRev.reverse();
    const summary = this.historyUseSummary
      ? this.#buildDroppedSummary(dropped, remaining)
      : "";
    return { selected, summary };
  }

  #composePromptWithHistory(prompt, history) {
    const cleanPrompt = this.#normalizeText(prompt);
    const turns = this.#sanitizeHistory(history);
    if (!turns.length) return cleanPrompt;

    const packed = this.#packHistory(turns, cleanPrompt);
    const lines = [];
    if (packed.summary) {
      lines.push(packed.summary);
    }
    for (const turn of packed.selected) {
      if (turn.role === "user") {
        lines.push(`User: ${turn.content}`);
      } else if (turn.role === "system") {
        lines.push(`Context: ${turn.content}`);
      } else {
        lines.push(`Assistant: ${turn.content}`);
      }
    }

    if (!lines.length) return cleanPrompt;
    lines.push(`User: ${cleanPrompt}`);
    lines.push("Assistant:");
    return `Conversation context:\n\n${lines.join("\n\n")}`;
  }

  #applyThinkingDirective(prompt, thinkMode) {
    const cleanPrompt = String(prompt ?? "").trim();
    const directive = thinkMode ? "/think" : "/no_think";
    if (!cleanPrompt) return directive;
    return `${directive}\n${cleanPrompt}`;
  }

  #maybeTuneSampling(temperature, topP, topK, repetitionPenalty, thinkMode) {
    let t = Number.isFinite(temperature) ? temperature : 0.0;
    let p = Number.isFinite(topP) ? topP : 1.0;
    let k = Number.isFinite(topK) ? topK : 20;
    let r = Number.isFinite(repetitionPenalty) ? repetitionPenalty : 1.0;
    void thinkMode;
    t = Math.max(0.0, Math.min(1.5, t));
    p = Math.max(0.05, Math.min(1.0, p));
    k = Math.max(1, Math.round(k));
    r = Math.max(1.0, Math.min(2.0, r));
    return { temperature: t, topP: p, topK: k, repetitionPenalty: r };
  }
}
