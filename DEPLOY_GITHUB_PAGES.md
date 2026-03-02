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

`model-q4_k_m.gguf` is ~379 MB and should not be committed directly to GitHub git history.

- GitHub git pushes reject files larger than 100 MB.
- The workflow expects a downloadable model URL in `MODEL_URL`.

Host model files somewhere public (for example a GitHub Release asset, HF, or object storage), then set:

- `MODEL_URL` (recommended): direct URL to `model-q4_k_m.gguf`
- `MODEL_BASE_URL` (optional): base URL that contains:
  - `config.json`
  - `generation_config.json`
  - `tokenizer.json`
  - `model-q4_k_m.gguf`
- `MODEL_SHA256` (optional): checksum for verification

If `MODEL_BASE_URL` is omitted, the script derives it from `MODEL_URL`.

## One-time GitHub setup

1. Create repo `neurovlm/neurolm`.
2. Push this project to `main` on:
   `git@github.com:neurovlm/neurolm.git`
3. In GitHub repo settings:
   - `Settings -> Pages -> Build and deployment -> Source`: select `GitHub Actions`.
4. In `Settings -> Secrets and variables -> Actions`, add:
   - `MODEL_URL` (or `MODEL_BASE_URL`)
   - `MODEL_BASE_URL` (optional but recommended)
   - `MODEL_SHA256` (optional)

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
