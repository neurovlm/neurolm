use std::collections::HashSet;
use std::io::Cursor;

use candle_core::{quantized::gguf_file, Device, Tensor, D};
use candle_transformers::models::quantized_qwen3::ModelWeights;
use js_sys::Function;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

const STREAM_TOKEN_BATCH: usize = 12;
const STREAM_FLUSH_INTERVAL_MS: f64 = 60.0;
const REPETITION_CONTEXT_WINDOW: usize = 256;

fn now_ms() -> f64 {
    if let Some(win) = web_sys::window() {
        if let Some(perf) = win.performance() {
            return perf.now();
        }
    }

    // Worker/global fallback.
    let global = js_sys::global();
    let perf = js_sys::Reflect::get(&global, &JsValue::from_str("performance")).ok();
    if let Some(perf_obj) = perf {
        let now_fn = js_sys::Reflect::get(&perf_obj, &JsValue::from_str("now")).ok();
        if let Some(f) = now_fn.and_then(|v| v.dyn_into::<Function>().ok()) {
            if let Ok(v) = f.call0(&perf_obj) {
                if let Some(ms) = v.as_f64() {
                    return ms;
                }
            }
        }
    }

    0.0
}

fn js_error<E: std::fmt::Display>(err: E) -> JsValue {
    JsValue::from_str(&err.to_string())
}

#[derive(Debug, Clone, Deserialize)]
struct QuantConfig {
    bits: Option<usize>,
    group_size: Option<usize>,
    quant_method: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelConfig {
    eos_token_id: Option<u32>,
    quantization_config: Option<QuantConfig>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct GenerationConfig {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repetition_penalty: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct ModelMeta {
    model: Option<ModelMetaModel>,
    runtime: Option<ModelMetaRuntime>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct ModelMetaModel {
    safetensors_file_mb: Option<f64>,
    compression_ratio_vs_bf16: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct ModelMetaRuntime {
    bits: Option<usize>,
    group_size: Option<usize>,
    quant_method: Option<String>,
    act_precision: Option<String>,
}

#[derive(Debug, Serialize)]
struct GenerationMetrics {
    prompt_tokens: usize,
    generated_tokens: usize,
    elapsed_ms: f64,
    first_token_ms: f64,
    tokens_per_sec: f64,
    model_memory_mb: f64,
    compression_ratio: f64,
    precision: String,
}

#[derive(Debug, Serialize)]
struct GenerationOutput {
    text: String,
    metrics: GenerationMetrics,
    mode: String,
}

#[wasm_bindgen]
pub struct NeuroSession {
    tokenizer: Option<Tokenizer>,
    config: Option<ModelConfig>,
    generation: GenerationConfig,
    meta: ModelMeta,
    eos_ids: Vec<u32>,
    rng: SmallRng,
    device: Device,
    model: Option<ModelWeights>,
}

#[wasm_bindgen]
impl NeuroSession {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            tokenizer: None,
            config: None,
            generation: GenerationConfig::default(),
            meta: ModelMeta::default(),
            eos_ids: Vec::new(),
            rng: SmallRng::seed_from_u64(0xA5A5_5A5A_0123_4567),
            device: Device::Cpu,
            model: None,
        }
    }

    pub fn configure(
        &mut self,
        config_json: &str,
        generation_json: &str,
        tokenizer_json: &str,
        meta_json: &str,
    ) -> Result<(), JsValue> {
        let config: ModelConfig = serde_json::from_str(config_json).map_err(js_error)?;
        let generation: GenerationConfig =
            serde_json::from_str(generation_json).map_err(js_error)?;
        let meta: ModelMeta = serde_json::from_str(meta_json).map_err(js_error)?;
        let tokenizer = Tokenizer::from_bytes(tokenizer_json.as_bytes()).map_err(js_error)?;

        let mut eos_ids = Vec::new();
        if let Some(eos) = config.eos_token_id {
            eos_ids.push(eos);
        }

        self.tokenizer = Some(tokenizer);
        self.config = Some(config);
        self.generation = generation;
        self.meta = meta;
        self.eos_ids = eos_ids;

        Ok(())
    }

    /// Loads a GGUF model blob into Candle quantized Qwen3 runtime.
    pub fn load_model(&mut self, model_bytes: &[u8]) -> Result<(), JsValue> {
        let owned = model_bytes.to_vec();
        let mut cursor = Cursor::new(owned);
        let content = gguf_file::Content::read(&mut cursor).map_err(js_error)?;
        let model =
            ModelWeights::from_gguf(content, &mut cursor, &self.device).map_err(js_error)?;
        self.model = Some(model);
        Ok(())
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        enable_thinking: bool,
        repetition_penalty: f32,
    ) -> Result<String, JsValue> {
        let output = self.generate_internal(
            prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            enable_thinking,
            repetition_penalty,
            None,
        )?;
        serde_json::to_string(&output).map_err(js_error)
    }

    #[wasm_bindgen(js_name = generate_stream)]
    pub fn generate_stream(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        enable_thinking: bool,
        repetition_penalty: f32,
        on_token: &Function,
    ) -> Result<String, JsValue> {
        let output = self.generate_internal(
            prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            enable_thinking,
            repetition_penalty,
            Some(on_token),
        )?;
        serde_json::to_string(&output).map_err(js_error)
    }
}

impl NeuroSession {
    fn generate_internal(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        enable_thinking: bool,
        repetition_penalty: f32,
        on_token: Option<&Function>,
    ) -> Result<GenerationOutput, JsValue> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| js_error("Tokenizer is not configured"))?;
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| js_error("Model config is not configured"))?;
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| js_error("Model is not loaded. Provide GGUF via load_model()."))?;

        if max_new_tokens == 0 {
            return Err(js_error("max_new_tokens must be >= 1"));
        }

        let default_temp = self.generation.temperature.unwrap_or(0.7);
        let default_top_p = self.generation.top_p.unwrap_or(0.95);
        let default_top_k = self.generation.top_k.unwrap_or(40);
        let default_repetition_penalty = self.generation.repetition_penalty.unwrap_or(1.0);

        let temperature = if temperature.is_finite() {
            temperature
        } else {
            default_temp
        };
        let top_p = if top_p.is_finite() && top_p > 0.0 && top_p <= 1.0 {
            top_p
        } else {
            default_top_p.clamp(0.01, 1.0)
        };
        let top_k = if top_k == 0 {
            default_top_k.max(1)
        } else {
            top_k.max(1)
        };
        let repetition_penalty = if repetition_penalty.is_finite() && repetition_penalty > 0.0 {
            repetition_penalty
        } else {
            default_repetition_penalty
        }
        .max(1.0)
        .min(2.0);

        // Force explicit thinking-mode behavior in the assistant prefill.
        // This mirrors models that expect reasoning inside <think>...</think>.
        let assistant_prefix = if enable_thinking {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n"
        };
        let chat_prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}",
            prompt, assistant_prefix
        );
        let encoded = tokenizer.encode(chat_prompt, true).map_err(js_error)?;
        let mut prompt_ids = encoded.get_ids().to_vec();
        if prompt_ids.is_empty() {
            prompt_ids.push(0);
        }

        model.clear_kv_cache();

        let start_ms = now_ms();
        let mut first_token_ms: Option<f64> = None;

        let prefill = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), &self.device)
            .map_err(js_error)?;

        let mut logits = model.forward(&prefill, 0).map_err(js_error)?;
        let mut position = prompt_ids.len();

        let mut generated_ids: Vec<u32> = Vec::new();
        let mut streamed_text = String::new();
        let mut stream_token_buffer: Vec<u32> = Vec::with_capacity(STREAM_TOKEN_BATCH);
        let mut last_stream_flush_ms = start_ms;
        let mut repetition_context: Vec<u32> = if repetition_penalty > 1.0 {
            let keep = prompt_ids.len().min(REPETITION_CONTEXT_WINDOW);
            prompt_ids[prompt_ids.len() - keep..].to_vec()
        } else {
            Vec::new()
        };

        for step in 0..max_new_tokens {
            let next_token = sample_from_logits(
                &mut self.rng,
                &logits,
                temperature,
                top_k,
                top_p,
                default_temp,
                repetition_penalty,
                if repetition_penalty > 1.0 {
                    Some(repetition_context.as_slice())
                } else {
                    None
                },
            )
            .map_err(js_error)?;

            if self.eos_ids.contains(&next_token) {
                break;
            }

            generated_ids.push(next_token);
            if repetition_penalty > 1.0 {
                repetition_context.push(next_token);
                if repetition_context.len() > REPETITION_CONTEXT_WINDOW {
                    let excess = repetition_context.len() - REPETITION_CONTEXT_WINDOW;
                    repetition_context.drain(0..excess);
                }
            }

            if step == 0 {
                first_token_ms = Some((now_ms() - start_ms).max(1.0));
            }

            if let Some(cb) = on_token {
                stream_token_buffer.push(next_token);
                let now = now_ms();
                let should_flush = stream_token_buffer.len() >= STREAM_TOKEN_BATCH
                    || (now - last_stream_flush_ms) >= STREAM_FLUSH_INTERVAL_MS;
                if should_flush {
                    flush_stream_tokens(
                        tokenizer,
                        cb,
                        &mut stream_token_buffer,
                        &mut streamed_text,
                    )?;
                    last_stream_flush_ms = now;
                }
            }

            let input =
                Tensor::from_vec(vec![next_token], (1, 1), &self.device).map_err(js_error)?;
            logits = model.forward(&input, position).map_err(js_error)?;
            position += 1;
        }

        let elapsed_ms = (now_ms() - start_ms).max(1.0);
        let text = tokenizer.decode(&generated_ids, true).map_err(js_error)?;

        if let Some(cb) = on_token {
            flush_stream_tokens(tokenizer, cb, &mut stream_token_buffer, &mut streamed_text)?;
            if text != streamed_text {
                let delta = if text.starts_with(&streamed_text) {
                    text[streamed_text.len()..].to_string()
                } else {
                    text.clone()
                };
                if !delta.is_empty() {
                    cb.call2(
                        &JsValue::NULL,
                        &JsValue::from_str(&delta),
                        &JsValue::UNDEFINED,
                    )
                    .map_err(|err| {
                        if err.is_null() || err.is_undefined() {
                            js_error("stream final callback failed")
                        } else {
                            err
                        }
                    })?;
                }
            }
        }

        let model_mb = self
            .meta
            .model
            .as_ref()
            .and_then(|m| m.safetensors_file_mb)
            .unwrap_or(0.0);
        let compression_ratio = self
            .meta
            .model
            .as_ref()
            .and_then(|m| m.compression_ratio_vs_bf16)
            .unwrap_or(0.0);

        let bits = self
            .meta
            .runtime
            .as_ref()
            .and_then(|r| r.bits)
            .or_else(|| config.quantization_config.as_ref().and_then(|q| q.bits))
            .unwrap_or(4);
        let group_size = self
            .meta
            .runtime
            .as_ref()
            .and_then(|r| r.group_size)
            .or_else(|| {
                config
                    .quantization_config
                    .as_ref()
                    .and_then(|q| q.group_size)
            })
            .unwrap_or(128);
        let quant_method = self
            .meta
            .runtime
            .as_ref()
            .and_then(|r| r.quant_method.clone())
            .or_else(|| {
                config
                    .quantization_config
                    .as_ref()
                    .and_then(|q| q.quant_method.clone())
            })
            .unwrap_or_else(|| "gguf".to_string());

        let generated_tokens = generated_ids.len();
        let safe_gen = generated_tokens.max(1);
        let tokens_per_sec = (safe_gen as f64) / (elapsed_ms / 1000.0);

        let precision = format!(
            "{} W{}A16 G{}",
            quant_method.to_uppercase(),
            bits,
            group_size
        );

        let metrics = GenerationMetrics {
            prompt_tokens: prompt_ids.len(),
            generated_tokens,
            elapsed_ms: round2(elapsed_ms),
            first_token_ms: round2(first_token_ms.unwrap_or(elapsed_ms)),
            tokens_per_sec: round2(tokens_per_sec),
            model_memory_mb: round2(model_mb),
            compression_ratio: round2(compression_ratio),
            precision,
        };

        Ok(GenerationOutput {
            text,
            metrics,
            mode: "candle-wasm-cpu-qwen3-gguf".to_string(),
        })
    }
}

fn flush_stream_tokens(
    tokenizer: &Tokenizer,
    cb: &Function,
    token_buffer: &mut Vec<u32>,
    streamed_text: &mut String,
) -> Result<(), JsValue> {
    if token_buffer.is_empty() {
        return Ok(());
    }
    let delta = tokenizer
        .decode(token_buffer.as_slice(), true)
        .map_err(js_error)?;
    token_buffer.clear();
    if delta.is_empty() {
        return Ok(());
    }
    streamed_text.push_str(&delta);
    cb.call2(
        &JsValue::NULL,
        &JsValue::from_str(&delta),
        &JsValue::UNDEFINED,
    )
    .map_err(|err| {
        if err.is_null() || err.is_undefined() {
            js_error("stream token callback failed")
        } else {
            err
        }
    })?;
    Ok(())
}

fn sample_from_logits(
    rng: &mut SmallRng,
    logits: &Tensor,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    default_temp: f32,
    repetition_penalty: f32,
    repetition_context: Option<&[u32]>,
) -> Result<u32, String> {
    if temperature <= 0.0 && repetition_penalty <= 1.0 {
        let arg = logits.argmax(D::Minus1).map_err(|e| e.to_string())?;
        let next = if arg.rank() == 0 {
            arg.to_scalar::<u32>().map_err(|e| e.to_string())?
        } else {
            let idx = arg
                .flatten_all()
                .map_err(|e| e.to_string())?
                .to_vec1::<u32>()
                .map_err(|e| e.to_string())?;
            idx.first()
                .copied()
                .ok_or_else(|| "Empty argmax output".to_string())?
        };
        return Ok(next);
    }

    let logits = logits.squeeze(0).map_err(|e| e.to_string())?;
    let mut raw = logits.to_vec1::<f32>().map_err(|e| e.to_string())?;
    if raw.is_empty() {
        return Err("Empty logits".to_string());
    }
    if repetition_penalty > 1.0 {
        if let Some(context) = repetition_context {
            apply_repetition_penalty_to_logits(raw.as_mut_slice(), repetition_penalty, context);
        }
    }

    if temperature <= 0.0 {
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (idx, &val) in raw.iter().enumerate() {
            if val > best_val {
                best_val = val;
                best_idx = idx;
            }
        }
        return Ok(best_idx as u32);
    }

    let temp = if temperature.is_finite() {
        temperature.max(1e-4)
    } else {
        default_temp.max(1e-4)
    };
    let keep = top_k.max(1).min(raw.len());

    let mut indexed: Vec<(usize, f32)> = Vec::with_capacity(keep);
    for (i, &raw_l) in raw.iter().enumerate() {
        let l = raw_l / temp;
        if indexed.len() < keep {
            indexed.push((i, l));
            if indexed.len() == keep {
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            continue;
        }
        if l <= indexed[keep - 1].1 {
            continue;
        }
        indexed[keep - 1] = (i, l);
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    let max_logit = indexed[0].1;
    let mut probs: Vec<(usize, f32)> = indexed
        .into_iter()
        .map(|(i, l)| (i, (l - max_logit).exp()))
        .collect();

    let mut sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Ok(probs[0].0 as u32);
    }

    for (_, p) in &mut probs {
        *p /= sum;
    }

    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let cutoff = top_p.clamp(0.01, 1.0);
    let mut cumulative = 0.0f32;
    let mut filtered = Vec::with_capacity(probs.len());
    for (idx, p) in probs {
        cumulative += p;
        filtered.push((idx, p));
        if cumulative >= cutoff {
            break;
        }
    }

    sum = filtered.iter().map(|(_, p)| *p).sum();
    if sum <= 0.0 {
        return Ok(filtered[0].0 as u32);
    }

    let mut r = rng.gen::<f32>() * sum;
    let mut fallback_idx = 0usize;
    for (idx, p) in filtered {
        fallback_idx = idx;
        r -= p;
        if r <= 0.0 {
            return Ok(idx as u32);
        }
    }

    Ok(fallback_idx as u32)
}

fn apply_repetition_penalty_to_logits(raw: &mut [f32], penalty: f32, context: &[u32]) {
    if raw.is_empty() || penalty <= 1.0 || context.is_empty() {
        return;
    }

    let mut seen: HashSet<u32> = HashSet::with_capacity(context.len());
    for &token in context.iter().rev() {
        if !seen.insert(token) {
            continue;
        }
        let idx = token as usize;
        if idx >= raw.len() {
            continue;
        }
        let logit = &mut raw[idx];
        if *logit >= 0.0 {
            *logit /= penalty;
        } else {
            *logit *= penalty;
        }
    }
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
