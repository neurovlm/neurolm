let wasmModule = null;
let session = null;
let modelLoaded = false;
const WASM_BUILD_TAG = "20260228-3";

async function ensureSession() {
  if (session) return session;
  wasmModule = await import(`./pkg/lm_wasm.js?v=${WASM_BUILD_TAG}`);
  await wasmModule.default();
  session = new wasmModule.NeuroSession();
  return session;
}

async function handleInit(payload) {
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
  for (const name of candidates) {
    try {
      const resp = await fetch(`${payload.modelBaseUrl}/${name}`);
      if (!resp.ok) {
        lastErr = `Failed to fetch ${name} (${resp.status})`;
        continue;
      }
      bytes = new Uint8Array(await resp.arrayBuffer());
      selected = name;
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
  s.load_model(bytes);
  modelLoaded = true;
  return { ready: true, modelFile: selected };
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
  const enableThinking = payload.enableThinking === true;

  const genArity = typeof session.generate === "function" ? session.generate.length : 0;
  const raw =
    genArity >= 6
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
  const enableThinking = payload.enableThinking === true;

  const onToken = (delta, text) => {
    self.postMessage({
      id,
      stream: true,
      event: "token",
      delta: String(delta ?? ""),
      text: String(text ?? ""),
    });
  };

  const streamArity = typeof session.generate_stream === "function" ? session.generate_stream.length : 0;
  const raw =
    streamArity >= 7
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
