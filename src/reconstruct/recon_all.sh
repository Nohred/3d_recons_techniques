#!/usr/bin/env bash
# Procesa una vez y entrena/exporta Nerfacto y Splatfacto con el mismo dataset.
# Uso: recon_all.sh [objeto ...]     (sin argumentos: todos los objetos con data/<objeto>/raw/orbit.mp4)
set -uo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib.sh"
run_pipeline "nerf splat" "$@"
