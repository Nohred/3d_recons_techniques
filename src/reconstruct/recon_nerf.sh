#!/usr/bin/env bash
# Procesa (si hace falta), entrena Nerfacto y exporta nube de puntos.
# Uso: recon_nerf.sh [objeto ...]     (sin argumentos: todos los objetos con data/<objeto>/raw/orbit.mp4)
set -uo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib.sh"
run_pipeline "nerf" "$@"
