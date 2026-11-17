#!/usr/bin/env bash
# Procesa (si hace falta), entrena Splatfacto y exporta gaussianas.
# Uso: recon_splat.sh [objeto ...]     (sin argumentos: todos los objetos con data/<objeto>/raw/orbit.mp4)
set -uo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib.sh"
run_pipeline "splat" "$@"
