# Deploy NeuroLM to GitHub Pages

This project is a static runtime (HTML + JS + WASM). GitHub Pages is a good fit.

## What this repo now includes

- Workflow: `.github/workflows/deploy-pages.yml`
- Build/stage script: `tools/build_pages_dist.sh`

The workflow:

1. Uses prebuilt WASM artifacts from `site/pkg`.
2. Stages static files into `dist/`.
3. Publishes `dist/` to GitHub Pages.

## Build WASM locally (before pushing)

From repo root:

```bash
cd /home/rph/efficient_ai/neurolm
./tools/build_wasm_local.sh
```

Then commit the generated files in `site/pkg` (including `lm_wasm_bg.wasm`).

## Important model note

`model-q4_k_m.gguf` is ~379 MB, so normal git push is blocked (>100 MB).
Use Git LFS for `model/*.gguf` (already configured in `.gitattributes`).

Track/add model with LFS:

```bash
cd /home/rph/efficient_ai/neurolm
git lfs install
git add .gitattributes model/model-q4_k_m.gguf
```

The workflow checks out with `lfs: true`, so CI receives the real GGUF binary.

Alternative (optional): host model files externally and set:

- `MODEL_URL`: direct URL to `model-q4_k_m.gguf`
- `MODEL_BASE_URL`: base URL with `config.json`, `generation_config.json`, `tokenizer.json`
- `MODEL_SHA256` (optional)

## One-time GitHub setup

1. Create repo `neurovlm/neurolm`.
2. Push this project to `main` on:
   `git@github.com:neurovlm/neurolm.git`
3. In GitHub repo settings:
   - `Settings -> Pages -> Build and deployment -> Source`: select `GitHub Actions`.
4. Optional only if not using repo/LFS model files: add Actions secrets
   - `MODEL_URL`
   - `MODEL_BASE_URL`
   - `MODEL_SHA256`

## Local git commands

Run from `/home/rph/efficient_ai/neurolm`:

```bash
# Only if this folder is not already its own git repo:
git init
git branch -M main

git remote add origin git@github.com:neurovlm/neurolm.git
git add .
git commit -m "Add GitHub Pages deployment pipeline"
git push -u origin main
```

If `origin` already exists, update it:

```bash
git remote set-url origin git@github.com:neurovlm/neurolm.git
```

## Optional local dry run

```bash
cd /home/rph/efficient_ai/neurolm
MODEL_URL="https://your-host/path/model-q4_k_m.gguf" ./tools/build_pages_dist.sh
python3 -m http.server 8008 --directory dist
```

To test exactly what CI does (no wasm compile in CI):

```bash
SKIP_WASM_BUILD=1 MODEL_URL="https://your-host/path/model-q4_k_m.gguf" ./tools/build_pages_dist.sh
```

Open:

- `http://127.0.0.1:8008/`

## Published URL

After Actions deploy succeeds, project pages will be:

- `https://neurovlm.github.io/neurolm/`
