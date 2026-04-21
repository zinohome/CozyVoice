#!/bin/bash
# CozyVoice Base Runtime 镜像构建脚本
# 构建 cozy-voice:latest 本地镜像
#
# 用法：
#   ./build.sh          # 构建 cozy-voice:latest
#   ./build.sh voice    # 同上

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${1:-all}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[BUILD]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

build_cozy_voice() {
    log "Building cozy-voice:latest ..."
    [ -f "$ROOT_DIR/Dockerfile" ] || err "Dockerfile not found: $ROOT_DIR/Dockerfile"
    docker build -t cozy-voice:latest "$ROOT_DIR"
    log "cozy-voice:latest built."
}

case "$TARGET" in
    all|voice|cozy-voice)
        build_cozy_voice
        ;;
    *)
        echo "Usage: $0 [all|voice]"
        exit 1
        ;;
esac

echo ""
log "Done! Custom images:"
docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" \
    | grep -E "^(cozy-voice|REPOSITORY)"
