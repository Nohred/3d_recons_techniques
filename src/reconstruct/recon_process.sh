#!/usr/bin/env bash
# Solo extrae frames y estima camaras (COLMAP).
# Uso: recon_process.sh [objeto ...]     (sin argumentos: todos los objetos con data/<objeto>/raw/orbit.mp4)
set -uo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib.sh"
run_pipeline "" "$@"
