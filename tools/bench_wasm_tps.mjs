#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';

import initWasm, { NeuroSession } from '../site/pkg/lm_wasm.js';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const modelDir = path.resolve(root, 'model');
const wasmPath = path.resolve(root, 'site/pkg/lm_wasm_bg.wasm');

const prompt = process.argv[2] || 'Describe neural oscillations';
const maxNewTokens = Number(process.argv[3] || 64);

await initWasm(await fs.readFile(wasmPath));

const session = new NeuroSession();
session.configure(
  await fs.readFile(path.join(modelDir, 'config.json'), 'utf8'),
  await fs.readFile(path.join(modelDir, 'generation_config.json'), 'utf8'),
  await fs.readFile(path.join(modelDir, 'tokenizer.json'), 'utf8'),
  '{}',
);

const modelBytes = new Uint8Array(await fs.readFile(path.join(modelDir, 'model-q4_k_m.gguf')));
console.time('load_model');
session.load_model(modelBytes);
console.timeEnd('load_model');

console.time('generate');
const genArity = typeof session.generate === 'function' ? session.generate.length : 0;
const raw =
  genArity >= 7
    ? session.generate(prompt, maxNewTokens, 0.0, 1.0, 20, false, 1.0)
    : session.generate(prompt, maxNewTokens, 0.0, 1.0, 20, false);
console.timeEnd('generate');

const out = JSON.parse(raw);
console.log('\nmetrics:', out.metrics);
console.log('\npreview:', String(out.text || '').slice(0, 240));
