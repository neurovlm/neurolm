#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
if [[ ! -d "${ROOT_DIR}/site" && -d "${ROOT_DIR}/neurolm/site" ]]; then
  ROOT_DIR="${ROOT_DIR}/neurolm"
fi
DIST_DIR="${1:-${ROOT_DIR}/dist}"

MODEL_DIR_DST="${DIST_DIR}/model"
MODEL_FILE_NAME="${MODEL_FILE_NAME:-model-q4_k_m.gguf}"
MODEL_URL="${MODEL_URL:-}"
MODEL_BASE_URL="${MODEL_BASE_URL:-}"
MODEL_SHA256="${MODEL_SHA256:-}"
SKIP_MODEL="${SKIP_MODEL:-0}"
SKIP_WASM_BUILD="${SKIP_WASM_BUILD:-0}"

required_model_files=(
  "config.json"
  "generation_config.json"
  "tokenizer.json"
)

required_wasm_pkg_files=(
  "lm_wasm.js"
  "lm_wasm_bg.wasm"
  "package.json"
)

check_prebuilt_wasm_pkg() {
  local missing=0
  for file in "${required_wasm_pkg_files[@]}"; do
    if [[ ! -f "${ROOT_DIR}/site/pkg/${file}" ]]; then
      echo "error: missing prebuilt WASM file: ${ROOT_DIR}/site/pkg/${file}" >&2
      missing=1
    fi
  done
  if [[ "${missing}" -ne 0 ]]; then
    echo "error: prebuilt WASM assets are required. Build locally first:" >&2
    echo "       cd ${ROOT_DIR} && ./tools/build_wasm_local.sh" >&2
    exit 1
  fi
}

strip_url_query_fragment() {
  local value="$1"
  value="${value%%\?*}"
  value="${value%%\#*}"
  printf '%s' "${value}"
}

derive_model_base_url() {
  local model_url_clean
  if [[ -n "${MODEL_BASE_URL}" ]]; then
    printf '%s' "${MODEL_BASE_URL%/}"
    return
  fi

  if [[ -n "${MODEL_URL}" ]]; then
    model_url_clean="$(strip_url_query_fragment "${MODEL_URL}")"
    printf '%s' "${model_url_clean%/*}"
    return
  fi

  printf '%s' ""
}

download_with_retry() {
  local src="$1"
  local dst="$2"
  curl --fail --location --retry 3 --retry-delay 3 "${src}" -o "${dst}"
}

ensure_not_lfs_pointer() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    echo "error: file not found: ${file}" >&2
    exit 1
  fi
  if head -n 1 "${file}" | grep -qx 'version https://git-lfs.github.com/spec/v1'; then
    echo "error: ${file} is a Git LFS pointer, not the actual model binary." >&2
    echo "       Run 'git lfs pull' locally, and in Actions ensure checkout uses 'lfs: true'." >&2
    exit 1
  fi
}

if [[ "${SKIP_WASM_BUILD}" == "1" ]]; then
  echo "==> SKIP_WASM_BUILD=1 set, using prebuilt site/pkg artifacts"
  check_prebuilt_wasm_pkg
else
  if [[ ! -d "${ROOT_DIR}/wasm" ]]; then
    echo "==> wasm/ directory not found at ${ROOT_DIR}/wasm; using prebuilt site/pkg artifacts"
    check_prebuilt_wasm_pkg
  else
    "${ROOT_DIR}/tools/build_wasm_local.sh"
    check_prebuilt_wasm_pkg
  fi
fi

MODEL_BASE_URL_EFFECTIVE="$(derive_model_base_url)"
MODEL_DIR_SRC=""
if [[ -d "${ROOT_DIR}/model" ]]; then
  MODEL_DIR_SRC="${ROOT_DIR}/model"
elif [[ -d "${ROOT_DIR}/site/model" ]]; then
  MODEL_DIR_SRC="${ROOT_DIR}/site/model"
fi

if [[ -z "${MODEL_DIR_SRC}" && -z "${MODEL_BASE_URL_EFFECTIVE}" && "${SKIP_MODEL}" != "1" ]]; then
  echo "error: model files are unavailable." >&2
  echo "       Provide one of:" >&2
  echo "       1) local model dir (${ROOT_DIR}/model or ${ROOT_DIR}/site/model)" >&2
  echo "       2) MODEL_URL (and optionally MODEL_BASE_URL)" >&2
  exit 1
fi

echo "==> Staging static site into ${DIST_DIR}"
rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}"
cp -R "${ROOT_DIR}/site" "${DIST_DIR}/site"
cp "${ROOT_DIR}/index.html" "${DIST_DIR}/index.html"
rm -f "${DIST_DIR}/site/pkg/.gitignore"

if [[ -f "${ROOT_DIR}/CNAME" ]]; then
  cp "${ROOT_DIR}/CNAME" "${DIST_DIR}/CNAME"
fi
touch "${DIST_DIR}/.nojekyll"

echo "==> Copying model metadata"
mkdir -p "${MODEL_DIR_DST}"
for file in "${required_model_files[@]}"; do
  if [[ -n "${MODEL_DIR_SRC}" && -f "${MODEL_DIR_SRC}/${file}" ]]; then
    cp "${MODEL_DIR_SRC}/${file}" "${MODEL_DIR_DST}/${file}"
  elif [[ -n "${MODEL_BASE_URL_EFFECTIVE}" ]]; then
    echo "==> Downloading ${file} from ${MODEL_BASE_URL_EFFECTIVE}"
    download_with_retry "${MODEL_BASE_URL_EFFECTIVE}/${file}" "${MODEL_DIR_DST}/${file}"
  else
    echo "error: required model file unavailable: ${file}" >&2
    echo "       Set MODEL_BASE_URL or include local model metadata files." >&2
    exit 1
  fi
done

if [[ "${SKIP_MODEL}" == "1" ]]; then
  echo "==> SKIP_MODEL=1 set, skipping GGUF file staging"
elif [[ -n "${MODEL_DIR_SRC}" && -f "${MODEL_DIR_SRC}/${MODEL_FILE_NAME}" ]]; then
  echo "==> Using local GGUF: ${MODEL_DIR_SRC}/${MODEL_FILE_NAME}"
  cp "${MODEL_DIR_SRC}/${MODEL_FILE_NAME}" "${MODEL_DIR_DST}/${MODEL_FILE_NAME}"
elif [[ -n "${MODEL_URL}" ]]; then
  echo "==> Downloading GGUF from MODEL_URL"
  download_with_retry "${MODEL_URL}" "${MODEL_DIR_DST}/${MODEL_FILE_NAME}"
elif [[ -n "${MODEL_BASE_URL_EFFECTIVE}" ]]; then
  echo "==> Downloading GGUF from MODEL_BASE_URL"
  download_with_retry "${MODEL_BASE_URL_EFFECTIVE}/${MODEL_FILE_NAME}" "${MODEL_DIR_DST}/${MODEL_FILE_NAME}"
else
  echo "error: GGUF file not found and MODEL_URL is not set." >&2
  if [[ -n "${MODEL_DIR_SRC}" ]]; then
    echo "       Expected local: ${MODEL_DIR_SRC}/${MODEL_FILE_NAME}" >&2
  fi
  exit 1
fi

if [[ "${SKIP_MODEL}" != "1" ]]; then
  ensure_not_lfs_pointer "${MODEL_DIR_DST}/${MODEL_FILE_NAME}"
fi

if [[ -n "${MODEL_SHA256}" && -f "${MODEL_DIR_DST}/${MODEL_FILE_NAME}" ]]; then
  echo "==> Verifying GGUF checksum"
  echo "${MODEL_SHA256}  ${MODEL_DIR_DST}/${MODEL_FILE_NAME}" | sha256sum --check
fi

echo "==> Build bundle ready"
du -sh "${DIST_DIR}"
