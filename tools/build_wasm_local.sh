#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
if [[ ! -d "${ROOT_DIR}/site" && -d "${ROOT_DIR}/neurolm/site" ]]; then
  ROOT_DIR="${ROOT_DIR}/neurolm"
fi

if [[ ! -d "${ROOT_DIR}/wasm" ]]; then
  echo "error: wasm source directory not found at ${ROOT_DIR}/wasm" >&2
  exit 1
fi

if ! command -v wasm-pack >/dev/null 2>&1; then
  echo "error: wasm-pack is not installed. Install it with: cargo install wasm-pack --locked" >&2
  exit 1
fi

echo "==> Building WASM package"
pushd "${ROOT_DIR}/wasm" >/dev/null
RUSTFLAGS='--cfg getrandom_backend="wasm_js"' wasm-pack build --target web --release --out-dir ../site/pkg
popd >/dev/null

cat >"${ROOT_DIR}/site/pkg/.gitignore" <<'EOF'
# Keep pkg scoped while allowing tracked WASM artifacts.
*
!lm_wasm.js
!lm_wasm_bg.wasm
!lm_wasm.d.ts
!lm_wasm_bg.wasm.d.ts
!package.json
EOF

echo "==> WASM artifacts ready in ${ROOT_DIR}/site/pkg"
