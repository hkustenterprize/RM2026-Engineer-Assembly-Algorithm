#!/usr/bin/env bash
# =============================================================================
# setup_blender_env.sh — 一键下载 Blender 4.5.4 LTS 并配置渲染依赖
#
# 功能:
#   1. 下载 Blender 4.5.4 LTS (Linux x64, tar.xz)
#   2. 解压到 BLENDER_INSTALL_DIR (默认 ~/blender)
#   3. 将 blender 可执行文件添加到 PATH (~/.bashrc 和 ~/.zshrc)
#   4. 用 Blender 内置 Python 3.11 安装渲染所需的 Python 包
#
# 用法:
#   bash setup_blender_env.sh
#
# 环境变量 (可覆盖默认值):
#   BLENDER_INSTALL_DIR   解压目标目录 (默认: ~/blender)
#   BLENDER_VERSION       版本号 (默认: 4.5.4)
# =============================================================================

set -euo pipefail

# ── 配置 ──────────────────────────────────────────────────────────────────────
BLENDER_VERSION="${BLENDER_VERSION:-4.5.4}"
BLENDER_PYTHON_MINOR="11"            # Blender 4.5 内置 Python 3.11
BLENDER_INSTALL_DIR="${BLENDER_INSTALL_DIR:-$HOME/blender}"

# Blender 下载 URL (官方镜像)
BLENDER_PKG="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
BLENDER_URL="https://mirrors.aliyun.com/blender/release/Blender${BLENDER_VERSION%.*}/${BLENDER_PKG}"
# 若阿里镜像速度慢, 可改用官方:
# BLENDER_URL="https://download.blender.org/release/Blender${BLENDER_VERSION%.*}/${BLENDER_PKG}"

# ── 颜色输出 ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── 步骤 1: 下载 ──────────────────────────────────────────────────────────────
info "Blender ${BLENDER_VERSION} — 开始安装"
info "安装目录: ${BLENDER_INSTALL_DIR}"

BLENDER_EXTRACTED="${BLENDER_INSTALL_DIR}/blender-${BLENDER_VERSION}-linux-x64"

if [[ -f "${BLENDER_EXTRACTED}/blender" ]]; then
    warn "Blender 已存在于 ${BLENDER_EXTRACTED}, 跳过下载和解压"
else
    mkdir -p "${BLENDER_INSTALL_DIR}"
    TMP_PKG="/tmp/${BLENDER_PKG}"

    if [[ -f "${TMP_PKG}" ]]; then
        warn "检测到已有下载文件 ${TMP_PKG}, 直接使用"
    else
        info "下载: ${BLENDER_URL}"
        curl -L --progress-bar --retry 3 -o "${TMP_PKG}" "${BLENDER_URL}" \
            || error "下载失败, 请检查网络或手动下载: ${BLENDER_URL}"
    fi

    info "解压至 ${BLENDER_INSTALL_DIR}..."
    tar -xJf "${TMP_PKG}" -C "${BLENDER_INSTALL_DIR}"
    rm -f "${TMP_PKG}"
    info "解压完成"
fi

BLENDER_BIN="${BLENDER_EXTRACTED}/blender"
[[ -f "${BLENDER_BIN}" ]] || error "未找到 blender 可执行文件: ${BLENDER_BIN}"

# ── 步骤 2: 添加到 PATH ──────────────────────────────────────────────────────
EXPORT_LINE="export PATH=\"${BLENDER_EXTRACTED}:\$PATH\"  # blender"

add_to_rc() {
    local rc_file="$1"
    if [[ -f "${rc_file}" ]]; then
        if grep -qF "${BLENDER_EXTRACTED}" "${rc_file}"; then
            warn "PATH 已包含 Blender 路径 (${rc_file}), 跳过"
        else
            echo "" >> "${rc_file}"
            echo "${EXPORT_LINE}" >> "${rc_file}"
            info "已将 Blender 路径写入 ${rc_file}"
        fi
    fi
}

add_to_rc "$HOME/.bashrc"
add_to_rc "$HOME/.zshrc"

# 当前 shell 立即生效
export PATH="${BLENDER_EXTRACTED}:${PATH}"
info "blender 路径: $(command -v blender)"

# ── 步骤 3: 定位 Blender 内置 Python ─────────────────────────────────────────
# Blender 4.x 内置 Python 位于 <blender_dir>/<major.minor>/python/bin/
BLENDER_PYTHON="${BLENDER_EXTRACTED}/${BLENDER_VERSION%.*}/python/bin/python3.${BLENDER_PYTHON_MINOR}"

if [[ ! -f "${BLENDER_PYTHON}" ]]; then
    # 自动探测 (兼容路径差异)
    BLENDER_PYTHON=$(find "${BLENDER_EXTRACTED}" -name "python3.*" -type f | head -1)
    [[ -n "${BLENDER_PYTHON}" ]] || error "未找到 Blender 内置 Python, 请手动指定"
    warn "使用探测到的 Python: ${BLENDER_PYTHON}"
fi

info "Blender Python: ${BLENDER_PYTHON}"
"${BLENDER_PYTHON}" --version

# ── 步骤 4: 安装 Python 依赖 ──────────────────────────────────────────────────
info "安装渲染依赖..."

# 确保 pip 可用 (Blender Python 通常已内置, 但有时需要 ensurepip)
"${BLENDER_PYTHON}" -m ensurepip --quiet 2>/dev/null || true
"${BLENDER_PYTHON}" -m pip install --quiet --upgrade pip

# 依赖列表
PACKAGES=(
    "opencv-python-headless"   # 径向畸变 cv2.remap + 图像处理
    "pyyaml"                   # YAML 配置文件解析
)

for pkg in "${PACKAGES[@]}"; do
    info "安装 ${pkg}..."
    "${BLENDER_PYTHON}" -m pip install --quiet "${pkg}"
done

# ── 验证 ──────────────────────────────────────────────────────────────────────
info "验证安装..."
"${BLENDER_PYTHON}" - <<'EOF'
import cv2, yaml, numpy as np
print(f"  cv2    {cv2.__version__}")
print(f"  yaml   {yaml.__version__}")
print(f"  numpy  {np.__version__}")
print("  ✓ 所有依赖验证通过")
EOF

echo ""
info "========================================================"
info " Blender ${BLENDER_VERSION} 安装完成!"
info ""
info " 可执行文件: ${BLENDER_BIN}"
info ""
info " 使用示例 (合成数据渲染):"
info "   cd cv/nn/blender"
info "   conda run -n mujoco-sim python scripts/render_parallel.py \\"
info "     --blender '${BLENDER_BIN}' \\"
info "     --n_images 100 --output_dir ./dataset --n_workers 4"
info ""
warn " 请重新打开终端或运行 'source ~/.bashrc' 使 PATH 生效"
info "========================================================"
