let wasmModule = null;
let session = null;
let modelLoaded = false;
const WASM_BUILD_TAG = "20260302-09";
const MODEL_CACHE_NAME = "neurolm-model-cache-v1";

function emitInitProgress(payload) {
  self.postMessage({
    event: "init-progress",
    payload: payload || {},
  });
}

function concatChunks(chunks, totalBytes) {
  const out = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

async function readModelFromCache(url) {
  if (typeof caches === "undefined") return null;
  try {
    const cache = await caches.open(MODEL_CACHE_NAME);
    const resp = await cache.match(url);
    if (!resp || !resp.ok) return null;
    const bytes = new Uint8Array(await resp.arrayBuffer());
    if (!bytes.length) return null;
    return bytes;
  } catch {
    return null;
  }
}

async function writeModelToCache(url, bytes) {
  if (typeof caches === "undefined") return;
  try {
    const cache = await caches.open(MODEL_CACHE_NAME);
    const headers = new Headers({
      "content-type": "application/octet-stream",
      "content-length": String(bytes.byteLength),
      "cache-control": "public, max-age=31536000, immutable",
    });
    await cache.put(
      url,
      new Response(bytes, {
        status: 200,
        headers,
      }),
    );
  } catch {
    // Cache persistence is best-effort only.
  }
}

async function fetchBytesWithProgress(url, modelFile) {
  const resp = await fetch(url, { cache: "default" });
  if (!resp.ok) {
    throw new Error(`Failed to fetch ${modelFile} (${resp.status})`);
  }
  const totalBytes = Number(resp.headers.get("content-length") || 0);
  if (!resp.body) {
    const raw = new Uint8Array(await resp.arrayBuffer());
    emitInitProgress({
      stage: "downloading",
      modelFile,
      loadedBytes: raw.byteLength,
      totalBytes: totalBytes || raw.byteLength,
      progress: 100,
    });
    return raw;
  }

  const reader = resp.body.getReader();
  const chunks = [];
  let loadedBytes = 0;
  let lastEmit = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;
    chunks.push(value);
    loadedBytes += value.length;
    const now = performance.now();
    if (now - lastEmit >= 120) {
      emitInitProgress({
        stage: "downloading",
        modelFile,
        loadedBytes,
        totalBytes,
        progress: totalBytes > 0 ? (loadedBytes / totalBytes) * 100 : NaN,
      });
      lastEmit = now;
    }
  }

  emitInitProgress({
    stage: "downloading",
    modelFile,
    loadedBytes,
    totalBytes,
    progress: totalBytes > 0 ? 100 : NaN,
  });
  return concatChunks(chunks, loadedBytes);
}

async function loadModelCandidate(modelBaseUrl, modelFile) {
  const modelUrl = `${modelBaseUrl}/${modelFile}`;

  const cached = await readModelFromCache(modelUrl);
  if (cached) {
    emitInitProgress({
      stage: "cache-hit",
      modelFile,
      source: "cache",
      loadedBytes: cached.byteLength,
      totalBytes: cached.byteLength,
      progress: 100,
    });
    return { bytes: cached, source: "cache" };
  }

  const bytes = await fetchBytesWithProgress(modelUrl, modelFile);
  await writeModelToCache(modelUrl, bytes);
  return { bytes, source: "network" };
}

async function ensureSession() {
  if (session) return session;
  wasmModule = await import(`./pkg/lm_wasm.js?v=${WASM_BUILD_TAG}`);
  await wasmModule.default();
  session = new wasmModule.NeuroSession();
  return session;
}

async function handleInit(payload) {
  emitInitProgress({
    stage: "init",
    message: "Preparing WASM runtime",
  });
  const s = await ensureSession();
  s.configure(
    JSON.stringify(payload.config),
    JSON.stringify(payload.generation),
    JSON.stringify(payload.tokenizer),
    JSON.stringify(payload.meta),
  );

  const candidates = [
    "model-q4_k_m.gguf",
    "model-q4_0.gguf",
    "model.gguf",
  ];
  let lastErr = null;
  let selected = null;
  let bytes = null;
  let source = "network";
  for (const name of candidates) {
    try {
      const loaded = await loadModelCandidate(payload.modelBaseUrl, name);
      bytes = loaded.bytes;
      selected = name;
      source = loaded.source;
      break;
    } catch (err) {
      lastErr = String(err);
    }
  }
  if (!bytes || !selected) {
    throw new Error(
      `Failed to load GGUF model. Tried: ${candidates.join(", ")}. Last error: ${lastErr || "unknown"}`
    );
  }
  emitInitProgress({
    stage: "loading-model",
    modelFile: selected,
    source,
    message: "Loading model into WASM CPU runtime",
  });
  s.load_model(bytes);
  modelLoaded = true;
  emitInitProgress({
    stage: "model-loaded",
    modelFile: selected,
    source,
    loadedBytes: bytes.byteLength,
    totalBytes: bytes.byteLength,
    progress: 100,
  });
  return { ready: true, modelFile: selected, modelSource: source, device: "cpu" };
}

function handleGenerate(payload) {
  if (!session || !modelLoaded) {
    throw new Error("WASM runtime not initialized");
  }

  const prompt = String(payload.prompt ?? "");
  const maxNewTokens = Number(payload.maxNewTokens ?? 128);
  const temperature = Number(payload.temperature ?? 0.6);
  const topP = Number(payload.topP ?? 0.95);
  const topK = Number(payload.topK ?? 20);
  const repetitionPenalty = Number(payload.repetitionPenalty ?? 1.0);
  const enableThinking = payload.enableThinking === true;

  const genArity = typeof session.generate === "function" ? session.generate.length : 0;
  const raw =
    genArity >= 7
      ? session.generate(
          prompt,
          maxNewTokens,
          temperature,
          topP,
          topK,
          enableThinking,
          repetitionPenalty,
        )
      : genArity >= 6
        ? session.generate(prompt, maxNewTokens, temperature, topP, topK, enableThinking)
      : session.generate(prompt, maxNewTokens, temperature, topP, topK);
  return JSON.parse(raw);
}

function handleGenerateStream(id, payload) {
  if (!session || !modelLoaded) {
    throw new Error("WASM runtime not initialized");
  }

  const prompt = String(payload.prompt ?? "");
  const maxNewTokens = Number(payload.maxNewTokens ?? 128);
  const temperature = Number(payload.temperature ?? 0.6);
  const topP = Number(payload.topP ?? 0.95);
  const topK = Number(payload.topK ?? 20);
  const repetitionPenalty = Number(payload.repetitionPenalty ?? 1.0);
  const enableThinking = payload.enableThinking === true;

  let pendingDelta = "";
  let lastFlushTs = 0;
  const flushTokenDelta = (force = false) => {
    if (!pendingDelta) return;
    const now = performance.now();
    if (!force && now - lastFlushTs < 45 && pendingDelta.length < 96) {
      return;
    }
    self.postMessage({
      id,
      stream: true,
      event: "token",
      delta: pendingDelta,
    });
    pendingDelta = "";
    lastFlushTs = now;
  };

  const onToken = (delta) => {
    const piece = String(delta ?? "");
    if (!piece) return;
    pendingDelta += piece;
    flushTokenDelta(false);
  };

  const streamArity = typeof session.generate_stream === "function" ? session.generate_stream.length : 0;
  const raw =
    streamArity >= 8
      ? session.generate_stream(
          prompt,
          maxNewTokens,
          temperature,
          topP,
          topK,
          enableThinking,
          repetitionPenalty,
          onToken,
        )
      : streamArity >= 7
        ? session.generate_stream(
            prompt,
            maxNewTokens,
            temperature,
            topP,
            topK,
            enableThinking,
            onToken,
          )
      : session.generate_stream(prompt, maxNewTokens, temperature, topP, topK, onToken);
  flushTokenDelta(true);
  return JSON.parse(raw);
}

self.onmessage = async (event) => {
  const { id, type, payload } = event.data || {};
  try {
    let result;
    if (type === "init") {
      result = await handleInit(payload || {});
    } else if (type === "generate") {
      result = handleGenerate(payload || {});
    } else if (type === "generate_stream") {
      result = handleGenerateStream(id, payload || {});
      self.postMessage({ id, stream: true, event: "done", result });
      return;
    } else {
      throw new Error(`Unknown worker message type: ${type}`);
    }
    self.postMessage({ id, ok: true, result });
  } catch (err) {
    const msg = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    if (type === "generate_stream") {
      self.postMessage({ id, stream: true, event: "error", error: msg });
    } else {
      self.postMessage({ id, ok: false, error: msg });
    }
  }
};
