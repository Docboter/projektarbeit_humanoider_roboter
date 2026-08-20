#!/usr/bin/env bash
# TL;DR: Gesourcte Bibliothek — löst GROOT_VERSION (1.6|1.7|auto) in Code-Baum, venv,
#   Modell-Repo und Namespace-Suffix auf; wortgleiche Kopie unter Simulation/scripts/.
# lib_groot_version.sh — Auswahl zwischen GR00T N1.6 und N1.7 (Modell + Code-Baum + venv).
#
# Wird von allen Launchern/Entrypoints GESOURCED (Training UND Simulation). Beide Images
# kopieren ihr scripts/-Verzeichnis nach /scripts/, daher liegt eine wortgleiche Kopie in
#   Training/scripts/lib_groot_version.sh      ← kanonische Fassung
#   Simulation/scripts/lib_groot_version.sh    ← Kopie (Build-Kontext ist Simulation/)
# Beide Kopien müssen identisch bleiben (`diff` ist Teil der Review-Checkliste).
#
# Warum: N1.6 (Eagle-Backbone, Python 3.10, Paket gr00t_n1d6) und N1.7 (Cosmos-Reason2-2B/
# Qwen3-VL, Python 3.12, Paket gr00t_n1d7) haben getrennte Code-Bäume und venvs
# (/app/Groot-1.6 bzw. /app/Groot-1.7) und nicht gegenseitig ladbare Checkpoints. Ein Image
# enthält BEIDE; dieses Skript wählt zur Laufzeit aus — N1.6 bleibt der Default.
#
# Eingang (Env):
#   GROOT_VERSION          1.6 | 1.7 | auto   (auch n1.6/N1.7/16/17/n16/n17 werden verstanden)
#                          Default: ${GROOT_VERSION_DEFAULT:-1.6}. `auto` ist nur sinnvoll,
#                          wenn ein Checkpoint vorliegt (Sim): groot_resolve_from_checkpoint.
#   GROOT_ROOT             optional expliziter Code-Baum. Die beiden Standardpfade
#                          /app/Groot-1.6 und /app/Groot-1.7 gelten NICHT als explizit: steht
#                          z. B. /app/Groot-1.6 aus einem Image-ENV im Environment, während
#                          GROOT_VERSION=1.7 gewählt ist, wird auf /app/Groot-1.7 umgestellt.
#   DATA_DIR               Default /data (für HF_HOME-Default)
#
# Ausgang (exportiert) nach groot_resolve:
#   GROOT_VERSION          normalisiert: "1.6" | "1.7"
#   GROOT_ROOT             /app/Groot-1.6 | /app/Groot-1.7 (oder expliziter Pfad)
#   GROOT_VENV_PY          $GROOT_ROOT/.venv/bin/python
#   GROOT_MODEL_NAME       GR00T-N1.6-3B | GR00T-N1.7-3B   (Verzeichnisname unter $DATA_DIR/models)
#   GROOT_MODEL_REPO       nvidia/GR00T-N1.6-3B | nvidia/GR00T-N1.7-3B
#   GROOT_BACKBONE_REPO    "" | nvidia/Cosmos-Reason2-2B  (N1.7 lädt das Backbone bei JEDEM
#                          Checkpoint-Laden vom HF-Hub nach → gated Repo, Zugang beantragen;
#                          liegt nach download_data.sh im HF-Cache, offline via HF_HUB_OFFLINE=1)
#   GROOT_MODEL_PKG        gr00t_n1d6 | gr00t_n1d7         (Modellpaket im Checkpoint, model_type
#                          in config.json: Gr00tN1d6 | Gr00tN1d7)
#   GROOT_NS_SUFFIX        "" | "_n17"  (Namespace-Suffix für Output-Dirs, damit N1.7-Checkpoints
#                          nicht mit N1.6-Läufen kollidieren: blockstacking → blockstacking_n17)
#   GROOT_TAG              n16 | n17    (kurzes Label für Log-Zeilen, Image-Tags, W&B-Tags)
#   HF_HOME                Default $DATA_DIR/hf_cache — persistent im Container-FS bzw. auf dem
#                          KISSKI-VAST-Mount, damit das 2-B-Backbone nicht bei jedem Start neu
#                          geladen wird. Ein bereits gesetztes HF_HOME wird respektiert.
#
# Funktionen:
#   groot_normalize_version <str>            → echo "1.6"|"1.7"|"auto", Exit 1 bei Unbekanntem
#   groot_resolve [version]                  → exportiert alles oben; bricht bei `auto` ab
#   groot_detect_version <checkpoint_dir>    → echo "1.6"|"1.7" aus config.json, Exit 1 sonst
#   groot_resolve_from_checkpoint <ckpt_dir> → bei GROOT_VERSION=auto: erkennen, dann resolve;
#                                              bei explizitem GROOT_VERSION: resolve + Warnung,
#                                              falls der Checkpoint offensichtlich nicht passt
#   groot_activate_venv                      → PATH/VIRTUAL_ENV auf $GROOT_ROOT/.venv (für
#                                              Skripte, die nacktes `python` rufen)
#   groot_summary                            → eine Zeile fürs Log

groot_normalize_version() {
    local v
    v="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
    case "$v" in
        1.6|n1.6|16|n16|1_6|n1d6|gr00t_n1d6|gr00tn1d6) echo "1.6" ;;
        1.7|n1.7|17|n17|1_7|n1d7|gr00t_n1d7|gr00tn1d7) echo "1.7" ;;
        auto|detect)                                    echo "auto" ;;
        "")                                             echo "${GROOT_VERSION_DEFAULT:-1.6}" ;;
        *)  echo "lib_groot_version: unbekannte GROOT_VERSION '$1' (erlaubt: 1.6, 1.7, auto)" >&2
            return 1 ;;
    esac
}

groot_resolve() {
    local v
    v="$(groot_normalize_version "${1:-${GROOT_VERSION:-}}")" || return 1
    if [[ "$v" == "auto" ]]; then
        echo "lib_groot_version: GROOT_VERSION=auto braucht einen Checkpoint" \
             "(groot_resolve_from_checkpoint <dir>) — oder GROOT_VERSION=1.6|1.7 setzen." >&2
        return 1
    fi
    export GROOT_VERSION="$v"
    local default_root
    case "$v" in
        1.6)
            default_root="/app/Groot-1.6"
            export GROOT_MODEL_NAME="GR00T-N1.6-3B"
            export GROOT_MODEL_REPO="nvidia/GR00T-N1.6-3B"
            export GROOT_BACKBONE_REPO=""
            export GROOT_MODEL_PKG="gr00t_n1d6"
            export GROOT_NS_SUFFIX=""
            export GROOT_TAG="n16"
            ;;
        1.7)
            default_root="/app/Groot-1.7"
            export GROOT_MODEL_NAME="GR00T-N1.7-3B"
            export GROOT_MODEL_REPO="nvidia/GR00T-N1.7-3B"
            export GROOT_BACKBONE_REPO="nvidia/Cosmos-Reason2-2B"
            export GROOT_MODEL_PKG="gr00t_n1d7"
            export GROOT_NS_SUFFIX="_n17"
            export GROOT_TAG="n17"
            ;;
    esac
    # GROOT_ROOT: expliziter Fremdpfad bleibt; die beiden Standardpfade folgen der Version.
    case "${GROOT_ROOT:-}" in
        ""|/app/Groot-1.6|/app/Groot-1.7|/app/Groot-1.6/|/app/Groot-1.7/)
            export GROOT_ROOT="$default_root" ;;
        *)  export GROOT_ROOT="${GROOT_ROOT%/}" ;;
    esac
    export GROOT_VENV_PY="$GROOT_ROOT/.venv/bin/python"
    export HF_HOME="${HF_HOME:-${DATA_DIR:-/data}/hf_cache}"
    return 0
}

# Liest model_type aus config.json: Gr00tN1d6 → 1.6, Gr00tN1d7 → 1.7.
# Akzeptiert das Checkpoint-Verzeichnis selbst oder ein Eltern-Verzeichnis mit checkpoint-*/.
groot_detect_version() {
    local dir="${1:-}" cfg="" c
    [[ -n "$dir" ]] || return 1
    for c in "$dir/config.json" "$dir"/checkpoint-*/config.json; do
        [[ -f "$c" ]] && { cfg="$c"; break; }
    done
    [[ -n "$cfg" ]] || return 1
    local mt
    mt="$(grep -o -E '"model_type"[[:space:]]*:[[:space:]]*"Gr00tN1d[67]"' "$cfg" | head -1)"
    case "$mt" in
        *Gr00tN1d6*) echo "1.6" ;;
        *Gr00tN1d7*) echo "1.7" ;;
        *) return 1 ;;
    esac
}

groot_resolve_from_checkpoint() {
    local ckpt="${1:-}" want detected
    want="$(groot_normalize_version "${GROOT_VERSION:-}")" || return 1
    detected="$(groot_detect_version "$ckpt" 2>/dev/null || true)"
    if [[ "$want" == "auto" ]]; then
        if [[ -z "$detected" ]]; then
            echo "lib_groot_version: GROOT_VERSION=auto, aber in '$ckpt' ist kein model_type" \
                 "Gr00tN1d6/Gr00tN1d7 erkennbar — falle auf N1.6 zurück" \
                 "(explizit: GROOT_VERSION=1.6|1.7)." >&2
            detected="1.6"
        fi
        groot_resolve "$detected"
    else
        groot_resolve "$want" || return 1
        if [[ -n "$detected" && "$detected" != "$GROOT_VERSION" ]]; then
            echo "lib_groot_version: WARNUNG — GROOT_VERSION=$GROOT_VERSION, aber der Checkpoint" \
                 "'$ckpt' ist ein N$detected-Checkpoint (model_type in config.json)." \
                 "N1.6- und N1.7-Checkpoints sind NICHT gegenseitig ladbar." >&2
        fi
    fi
}

groot_activate_venv() {
    [[ -n "${GROOT_ROOT:-}" ]] || groot_resolve || return 1
    export VIRTUAL_ENV="$GROOT_ROOT/.venv"
    # vorhandene /app/Groot-*/.venv/bin-Einträge aus PATH entfernen, dann unsere voranstellen
    local cleaned="" p
    IFS=':' read -r -a _parts <<< "${PATH:-}"
    for p in "${_parts[@]}"; do
        case "$p" in
            /app/Groot-1.6/.venv/bin|/app/Groot-1.7/.venv/bin) ;;
            "") ;;
            *) cleaned="${cleaned:+$cleaned:}$p" ;;
        esac
    done
    export PATH="$VIRTUAL_ENV/bin${cleaned:+:$cleaned}"
}

groot_summary() {
    printf 'GR00T N%s  root=%s  modell=%s  pkg=%s%s\n' \
        "${GROOT_VERSION:-?}" "${GROOT_ROOT:-?}" "${GROOT_MODEL_REPO:-?}" "${GROOT_MODEL_PKG:-?}" \
        "${GROOT_BACKBONE_REPO:+  backbone=$GROOT_BACKBONE_REPO (gated, HF_HOME=$HF_HOME)}"
}
