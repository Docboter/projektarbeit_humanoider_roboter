#!/usr/bin/env bash
# lib_split.sh — Train/Test-Split für die Finetuning-Läufe.
#
# Gemeinsame Logik von run_finetuning.sh und run_finetuning_vision.sh. Wird
# gesourct, nicht ausgeführt:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/lib_split.sh"
#     split_apply "$INFO_JSON" "$TRAIN_TEST_SPLIT" "$TRAIN_SPLIT_RATIO" "$OUTPUT_DIR"
#
# Hintergrund: Der Split-Filter (_apply_split_filter in
# gr00t/data/dataset/lerobot_episode_loader.py) liest den Bereich aus
# meta/info.json; der Trainings-Datensatz fragt fest split="train" ab
# (factory.py). Wir müssen also nur den splits-Eintrag in info.json setzen.
#
# WICHTIG — das ist die Voraussetzung für checkpoint_sweep.py: Ohne
# zurückgehaltene Episoden gibt es hinterher keine Validierungs-Zahl, und die
# Checkpoint-Auswahl bleibt blind (so geschehen in Lauf 1 und Lauf 2).

# Helfer nur definieren, wenn das aufrufende Skript sie nicht schon hat.
declare -F log  >/dev/null || log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
declare -F err  >/dev/null || err()  { printf '\033[1;31m!! \033[0m%s\n' "$*" >&2; }
declare -F warn >/dev/null || warn() { printf '\033[1;33m ! \033[0m%s\n' "$*"; }

# split_apply <info.json> <enabled 0|1> <ratio> [record_dir]
#
# enabled=1 → info.json bekommt {"train": "0:n", "test": "n:total"}; zusätzlich
#             wird (falls record_dir gesetzt) split.json daneben abgelegt, damit
#             später nachvollziehbar ist, welche Episoden zurückgehalten wurden.
# enabled=0 → ein evtl. vorhandener test-Split wird auf den vollen Datensatz
#             zurückgesetzt, damit ein Folgelauf reproduzierbar alles sieht.
split_apply() {
    local info_json="$1" enabled="$2" ratio="$3" record_dir="${4:-}"

    if [[ "$enabled" == "1" ]]; then
        if [[ ! -f "$info_json" ]]; then
            err "TRAIN_TEST_SPLIT=1, aber info.json fehlt: $info_json"
            return 1
        fi
        log "TRAIN_TEST_SPLIT=1 — setze Split (Ratio=$ratio) in info.json"
        python - "$info_json" "$ratio" "$record_dir" <<'PY'
import json, sys, os
info_path, ratio, record_dir = sys.argv[1], float(sys.argv[2]), sys.argv[3]
with open(info_path) as f:
    info = json.load(f)
total = int(info.get("total_episodes") or 0)
if total <= 0:
    sys.exit(f"info.json hat kein gueltiges total_episodes ({total}).")
n_train = int(total * ratio)
if not (0 < n_train < total):
    sys.exit(f"Ungueltiger Split: ratio={ratio} -> n_train={n_train} von {total}.")
splits = {"train": f"0:{n_train}", "test": f"{n_train}:{total}"}
info["splits"] = splits
with open(info_path, "w") as f:
    json.dump(info, f, indent=4)
print(f"[split] train=0:{n_train}  test={n_train}:{total}  (gesamt {total} Episoden)")
if record_dir:
    os.makedirs(record_dir, exist_ok=True)
    record = {
        "dataset_info_json": info_path,
        "total_episodes": total,
        "train_split_ratio": ratio,
        "splits": splits,
        "test_episode_count": total - n_train,
        "test_episode_index_range": [n_train, total],
    }
    record_path = os.path.join(record_dir, "split.json")
    with open(record_path, "w") as f:
        json.dump(record, f, indent=4)
    print(f"[split] Protokoll geschrieben: {record_path}")
PY
        log "Split aktiv: Test-Episoden werden NICHT mittrainiert."
        log "Nach dem Lauf: checkpoint_sweep.py wertet genau diese Episoden aus."
        return 0
    fi

    # enabled != 1 → ggf. zurücksetzen.
    if [[ -f "$info_json" ]] && grep -q '"test"' "$info_json" 2>/dev/null; then
        warn "TRAIN_TEST_SPLIT=0, aber info.json enthält einen test-Split — setze auf vollen Datensatz zurück."
        python - "$info_json" <<'PY'
import json, sys
info_path = sys.argv[1]
with open(info_path) as f:
    info = json.load(f)
total = int(info.get("total_episodes") or 0)
info["splits"] = {"train": f"0:{total}"}
with open(info_path, "w") as f:
    json.dump(info, f, indent=4)
print(f"[split] zurückgesetzt: train=0:{total} (voller Datensatz)")
PY
        if [[ -n "$record_dir" && -f "$record_dir/split.json" ]]; then
            rm -f "$record_dir/split.json"
            warn "Veraltetes split.json entfernt: $record_dir/split.json"
        fi
    fi
    warn "Ohne Split gibt es keine zurückgehaltenen Episoden — die Checkpoint-Auswahl"
    warn "bleibt danach blind (checkpoint_sweep.py hat nichts zum Validieren)."
    return 0
}
