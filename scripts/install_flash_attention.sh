#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/third_party/flash-attention/VERSION"
install_root="${FLASH_ATTN_INSTALL_ROOT:-${repo_root}/.deps/flash-attention}"

if [[ ! -d "${install_root}/.git" ]]; then
  git clone https://github.com/Dao-AILab/flash-attention.git "${install_root}"
fi
git -C "${install_root}" fetch origin "${base_commit}"
git -C "${install_root}" checkout --detach "${base_commit}"
git -C "${install_root}" submodule update --init --recursive
git -C "${install_root}/csrc/cutlass" checkout --detach "${cutlass_commit}"
if git -C "${install_root}" apply --reverse --check "${repo_root}/third_party/flash-attention/pair-bias.patch" 2>/dev/null; then
  git -C "${install_root}" apply --reverse "${repo_root}/third_party/flash-attention/pair-bias.patch"
fi
git -C "${install_root}" apply --check "${repo_root}/third_party/flash-attention/pair-bias.patch"
git -C "${install_root}" apply "${repo_root}/third_party/flash-attention/pair-bias.patch"

python -m pip install --no-build-isolation "${install_root}/hopper"
python - <<'PY'
import flash_attn_interface

required = (
    "flash_attn_varlen_func",
    "flash_attn_pair_bias_func",
    "flash_attn_compact_pair_bias_func",
)
missing = [name for name in required if not hasattr(flash_attn_interface, name)]
if missing:
    raise RuntimeError(f"custom FlashAttention installation lacks: {', '.join(missing)}")
print("custom FlashAttention symbols verified")
PY
