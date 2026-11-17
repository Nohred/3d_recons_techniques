#!/usr/bin/env bash
# Genera una nube densa de fotogrametría con COLMAP por objeto.
#
# Uso:
#   recon_photogrammetry.sh [objeto ...]
#   MAX_OBJECTS=3 recon_photogrammetry.sh
#
# Sin argumentos procesa objetos en orden lexicográfico. Solo considera
# objetos con processed/images y processed/colmap/sparse/0.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

# ================= CONFIGURACION EDITABLE =================
MAX_OBJECTS="${MAX_OBJECTS:-10}"                 # 0 = todos los candidatos
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-1300}"
PATCH_MATCH_GEOM_CONSISTENCY="${PATCH_MATCH_GEOM_CONSISTENCY:-true}"
FORCE_PHOTOGRAMMETRY="${FORCE_PHOTOGRAMMETRY:-0}"
# ==========================================================

check_photogrammetry_environment() {
    [[ -d "$DATA_DIR" ]] || die "No existe DATA_DIR: $DATA_DIR"
    if [[ "$DRY_RUN" != 1 ]]; then
        command -v colmap >/dev/null || die "colmap no esta en PATH"
    fi
    command -v flock >/dev/null || die "flock no esta disponible"
    [[ "$MAX_OBJECTS" =~ ^[0-9]+$ ]] || die "MAX_OBJECTS debe ser un entero >= 0"
    [[ "$MAX_IMAGE_SIZE" =~ ^[1-9][0-9]*$ ]] || die "MAX_IMAGE_SIZE debe ser positivo"
    if [[ "$DRY_RUN" != 1 ]]; then
        mkdir -p "$REPORT_DIR"
        exec 9>"$PROJECT_ROOT/.recon.lock"
        flock -n 9 || die "Ya hay otra reconstruccion usando este proyecto"
    fi
}

discover_photogrammetry_objects() {
    local obj dir
    if [[ $# -gt 0 ]]; then
        for obj in "$@"; do
            obj="${obj%.usdz}"
            dir="$DATA_DIR/$obj/reconstruction/processed"
            if [[ ! -d "$dir/images" || ! -d "$dir/colmap/sparse/0" ]]; then
                log "OMITIDO $obj: falta processed/images o processed/colmap/sparse/0" >&2
                continue
            fi
            printf '%s\n' "$obj"
        done
        return 0
    fi
    for dir in "$DATA_DIR"/*/reconstruction/processed/; do
        [[ -d "$dir/images" && -d "$dir/colmap/sparse/0" ]] || continue
        basename "$(dirname "$(dirname "$dir")")"
    done | sort
}

stage_photogrammetry() {
    local obj="$1"
    local base="$DATA_DIR/$obj/reconstruction"
    local processed="$base/processed"
    local input_images="$processed/images"
    local sparse="$processed/colmap/sparse/0"
    local output="$base/exports/photogrammetry"
    local dense="$output/dense"
    local fused="$output/fused.ply"
    local logfile="$base/logs/photogrammetry.log"

    if [[ -s "$fused" && "$FORCE_PHOTOGRAMMETRY" != 1 ]]; then
        STAGE_STATUS=skipped
        STAGE_DETAIL="fused.ply ya existe"
        return 0
    fi
    if [[ ! -d "$input_images" || ! -d "$sparse" ]]; then
        STAGE_STATUS=skipped
        STAGE_DETAIL="falta processed/images o processed/colmap/sparse/0"
        return 0
    fi

    if [[ "$DRY_RUN" != 1 ]]; then
        mkdir -p "$output" "$base/logs"
    fi
    run_logged "$logfile" colmap image_undistorter \
        --image_path "$input_images" \
        --input_path "$sparse" \
        --output_path "$dense" \
        --output_type COLMAP \
        --max_image_size "$MAX_IMAGE_SIZE" \
        || { STAGE_DETAIL="image_undistorter fallo"; return 1; }

    run_logged "$logfile" colmap patch_match_stereo \
        --workspace_path "$dense" \
        --workspace_format COLMAP \
        --PatchMatchStereo.geom_consistency "$PATCH_MATCH_GEOM_CONSISTENCY" \
        || { STAGE_DETAIL="patch_match_stereo fallo"; return 1; }

    run_logged "$logfile" colmap stereo_fusion \
        --workspace_path "$dense" \
        --workspace_format COLMAP \
        --input_type geometric \
        --output_path "$fused" \
        || { STAGE_DETAIL="stereo_fusion fallo"; return 1; }

    [[ -s "$fused" ]] || {
        STAGE_DETAIL="COLMAP no genero fused.ply"
        return 1
    }
    STAGE_DETAIL="fused=$fused"
}

main() {
    check_photogrammetry_environment
    local requested=()
    mapfile -t requested < <(discover_photogrammetry_objects "$@")
    [[ ${#requested[@]} -gt 0 ]] || die "No hay objetos con COLMAP disponible"

    local objects=()
    local obj
    for obj in "${requested[@]}"; do
        objects+=("$obj")
        [[ "$MAX_OBJECTS" -eq 0 || ${#objects[@]} -ge "$MAX_OBJECTS" ]] && break
    done
    log "Objetos de fotogrametria (${#objects[@]}): ${objects[*]}"

    local failed=()
    for obj in "${objects[@]}"; do
        if timed_stage "$obj" photogrammetry stage_photogrammetry "$obj"; then
            :
        else
            failed+=("$obj")
        fi
    done
    log "Fotogrametria terminada. Fallos: ${failed[*]:-ninguno}"
    [[ ${#failed[@]} -eq 0 ]]
}

main "$@"
