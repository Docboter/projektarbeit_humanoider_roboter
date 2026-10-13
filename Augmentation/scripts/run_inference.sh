#!/usr/bin/env bash
# TL;DR: Duenner Wrapper um cosmos-transfer2.5s examples/inference.py — ein Spec, ein Output-Ordner.
# run_inference.sh <spec.json> <out_dir>
#
# Eigenstaendig aufrufbar (auch per `docker exec` fuers Debugging einzelner Items), damit man
# nicht immer die ganze entrypoint.sh-Schleife anstossen muss.
#
# NUM_GPU=1 (default) -> einfacher python-Aufruf. NUM_GPU>1 -> torchrun-Modell-Split
# (Cosmos' eigener Variablenname, siehe Env-Var-Tabelle in docs/augmentation/anleitung.md).
# AUGMENT_MODEL_VARIANT waehlt Modell+Modalitaet via --model (gueltige Werte laut
# cosmos_transfer2/config.py::MODEL_CHECKPOINTS: depth, edge, seg, vis — "edge/distilled"
# NUR wenn COSMOS_EXPERIMENTAL_CHECKPOINTS=1 gesetzt ist, sonst "invalid choice").

set -euo pipefail

SPEC_PATH="${1:?Usage: run_inference.sh <spec.json> <out_dir>}"
OUT_DIR="${2:?Usage: run_inference.sh <spec.json> <out_dir>}"

NUM_GPU="${NUM_GPU:-1}"
AUGMENT_MODEL_VARIANT="${AUGMENT_MODEL_VARIANT:-edge}"
AUGMENT_DISABLE_GUARDRAILS="${AUGMENT_DISABLE_GUARDRAILS:-1}"

# Der distillierte Modell-Key ist in cosmos-transfer2.5 hinter einem experimentellen Flag
# versteckt (cosmos_transfer2/_src/imaginaire/flags.py: EXPERIMENTAL_CHECKPOINTS liest
# COSMOS_EXPERIMENTAL_CHECKPOINTS). Ohne das schlaegt --model=edge/distilled mit
# "invalid choice" fehl — hier automatisch gesetzt, damit AUGMENT_MODEL_VARIANT=edge/distilled
# nicht zusaetzlich diese zweite Variable erfordert.
if [[ "$AUGMENT_MODEL_VARIANT" == *distilled* ]]; then
    export COSMOS_EXPERIMENTAL_CHECKPOINTS="${COSMOS_EXPERIMENTAL_CHECKPOINTS:-1}"
fi

# Guardrails laden per Default NOCH EIN gated HF-Repo (nvidia/Cosmos-Guardrail1, eigene
# Lizenz-Zustimmung noetig, unabhaengig von Cosmos-Transfer2.5-2B) — fuer diese interne
# Trainingsdaten-Pipeline (eigene, bereits abgenommene Robotervideos) unnoetig. Default:
# aus. AUGMENT_DISABLE_GUARDRAILS=0 schaltet sie wieder ein (dann vorher zusaetzlich
# https://huggingface.co/nvidia/Cosmos-Guardrail1 akzeptieren).
GUARDRAIL_ARGS=()
if [[ "$AUGMENT_DISABLE_GUARDRAILS" == "1" ]]; then
    GUARDRAIL_ARGS=(--disable-guardrails)
fi

cd /app/cosmos-transfer2.5

if [[ "$NUM_GPU" -gt 1 ]]; then
    torchrun --nproc_per_node="$NUM_GPU" --master_port="${MASTER_PORT:-12341}" \
        examples/inference.py -i "$SPEC_PATH" -o "$OUT_DIR" --model="$AUGMENT_MODEL_VARIANT" \
        "${GUARDRAIL_ARGS[@]}"
else
    python examples/inference.py -i "$SPEC_PATH" -o "$OUT_DIR" --model="$AUGMENT_MODEL_VARIANT" \
        "${GUARDRAIL_ARGS[@]}"
fi
