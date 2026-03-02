#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${1:-${ROOT_DIR}/dist}"

MODEL_DIR_SRC="${ROOT_DIR}/model"
MODEL_DIR_DST="${DIST_DIR}/model"
MODEL_FILE_NAME="${MODEL_FILE_NAME:-model-q4_k_m.gguf}"
MODEL_URL="${MODEL_URL:-}"
MODEL_SHA256="${MODEL_SHA256:-}"
SKIP_MODEL="${SKIP_MODEL:-0}"

required_model_files=(
  "config.json"
  "generation_config.json"
  "tokenizer.json"
)

if ! command -v wasm-pack >/dev/null 2>&1; then
  echo "error: wasm-pack is not installed. Install it with: cargo install wasm-pack --locked" >&2
  exit 1
fi

echo "==> Building WASM package"
pushd "${ROOT_DIR}/wasm" >/dev/null
RUSTFLAGS='--cfg getrandom_backend="wasm_js"' wasm-pack build --target web --release --out-dir ../site/pkg
popd >/dev/null

echo "==> Staging static site into ${DIST_DIR}"
rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}"
cp -R "${ROOT_DIR}/site" "${DIST_DIR}/site"
cp "${ROOT_DIR}/index.html" "${DIST_DIR}/index.html"

if [[ -f "${ROOT_DIR}/CNAME" ]]; then
  cp "${ROOT_DIR}/CNAME" "${DIST_DIR}/CNAME"
fi
touch "${DIST_DIR}/.nojekyll"

echo "==> Copying model metadata"
mkdir -p "${MODEL_DIR_DST}"
for file in "${required_model_files[@]}"; do
  if [[ ! -f "${MODEL_DIR_SRC}/${file}" ]]; then
    echo "error: required model file missing: ${MODEL_DIR_SRC}/${file}" >&2
    exit 1
  fi
  cp "${MODEL_DIR_SRC}/${file}" "${MODEL_DIR_DST}/${file}"
done

if [[ "${SKIP_MODEL}" == "1" ]]; then
  echo "==> SKIP_MODEL=1 set, skipping GGUF file staging"
elif [[ -f "${MODEL_DIR_SRC}/${MODEL_FILE_NAME}" ]]; then
  echo "==> Using local GGUF: ${MODEL_DIR_SRC}/${MODEL_FILE_NAME}"
  cp "${MODEL_DIR_SRC}/${MODEL_FILE_NAME}" "${MODEL_DIR_DST}/${MODEL_FILE_NAME}"
elif [[ -n "${MODEL_URL}" ]]; then
  echo "==> Downloading GGUF from MODEL_URL"
  curl --fail --location --retry 3 --retry-delay 3 "${MODEL_URL}" -o "${MODEL_DIR_DST}/${MODEL_FILE_NAME}"
else
  echo "error: GGUF file not found and MODEL_URL is not set." >&2
  echo "       Expected local: ${MODEL_DIR_SRC}/${MODEL_FILE_NAME}" >&2
  exit 1
fi

if [[ -n "${MODEL_SHA256}" && -f "${MODEL_DIR_DST}/${MODEL_FILE_NAME}" ]]; then
  echo "==> Verifying GGUF checksum"
  echo "${MODEL_SHA256}  ${MODEL_DIR_DST}/${MODEL_FILE_NAME}" | sha256sum --check
fi

echo "==> Build bundle ready"
du -sh "${DIST_DIR}"
