#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
if [[ ! -d "${ROOT_DIR}/site" && -d "${ROOT_DIR}/neurolm/site" ]]; then
  ROOT_DIR="${ROOT_DIR}/neurolm"
fi

PORT="8008"
BUILD_ONLY="0"
SERVE_ONLY="0"

usage() {
  cat <<'EOF'
Usage:
  ./tools/build_and_test_local.sh [port]
  ./tools/build_and_test_local.sh --build-only [port]
  ./tools/build_and_test_local.sh --serve-only [port]

Examples:
  ./tools/build_and_test_local.sh
  ./tools/build_and_test_local.sh 8010
  ./tools/build_and_test_local.sh --build-only
  ./tools/build_and_test_local.sh --serve-only 8008
EOF
}

for arg in "$@"; do
  case "$arg" in
    --build-only)
      BUILD_ONLY="1"
      ;;
    --serve-only)
      SERVE_ONLY="1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ "$arg" =~ ^[0-9]+$ ]]; then
        PORT="$arg"
      else
        echo "error: unknown argument '$arg'" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

if [[ "${BUILD_ONLY}" == "1" && "${SERVE_ONLY}" == "1" ]]; then
  echo "error: use only one of --build-only or --serve-only" >&2
  exit 1
fi

if [[ "${SERVE_ONLY}" != "1" ]]; then
  echo "==> Building WASM package"
  "${ROOT_DIR}/tools/build_wasm_local.sh"

  echo "==> Building static dist bundle (chat route)"
  APP_ROUTE_DIR=chat SKIP_WASM_BUILD=1 "${ROOT_DIR}/tools/build_pages_dist.sh"
fi

echo "==> Dist ready: ${ROOT_DIR}/dist"
echo "    Root: http://127.0.0.1:${PORT}/"
echo "    Chat: http://127.0.0.1:${PORT}/chat/"

if [[ "${BUILD_ONLY}" == "1" ]]; then
  echo "==> Build-only mode complete (server not started)."
  exit 0
fi

echo "==> Starting local server on 127.0.0.1:${PORT}"
exec python3 -m http.server "${PORT}" --bind 127.0.0.1 --directory "${ROOT_DIR}/dist"
