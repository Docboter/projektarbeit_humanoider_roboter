#!/usr/bin/env python3
# TL;DR: Läuft im GR00T-venv: liest Pro-Gruppe State/Action-Dimensionen aus Checkpoint-Statistiken.
"""
Liest die Pro-Gruppe-State-/Action-Dimensionen eines GR00T-Checkpoints für ein
Embodiment aus den Normalisierungs-Statistiken des Prozessors und schreibt sie
als JSON. Wird im **GR00T-venv** ausgeführt (hat Zugriff auf `transformers` +
`gr00t`), bevor der Isaac-Lab-Sim-Client (anderes Python) startet.

Hintergrund: Der Gr00tSimPolicyWrapper erwartet pro State-Gruppe ein Array der
korrekten Dimension D. D ist NICHT in der Modality-Config hinterlegt, sondern
ergibt sich aus den Statistiken (Länge der mean/std-Listen) im Checkpoint.
Statt D hartzukodieren, lesen wir es hier zur Laufzeit aus.

Versionsunabhängig, weil hier weder `gr00t` noch `EmbodimentTag` importiert wird: der
Prozessor kommt über `AutoProcessor` + trust_remote_code aus dem Checkpoint selbst, und
`--embodiment-tag` ist einfach der Schlüssel in dessen Statistiken. Zu beachten ist nur,
dass derselbe NAME je nach Version einen anderen WERT hat — N1.6 `unitree_g1`, N1.7
`unitree_g1_full_body_with_waist_height_nav_cmd`. Der Baseline-Entrypoint, der dieses
Skript ruft, ist aus genau diesem Grund auf N1.6 beschränkt.

Verwendung:
    python dump_unitree_g1_dims.py \
        --model-path /data/checkpoints/GR00T-N1.6-3B \
        --embodiment-tag unitree_g1 \
        --out /data/g1_baseline_dims.json
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Dump GR00T per-group state/action dims")
    ap.add_argument("--model-path", required=True, help="Pfad zum Checkpoint-Verzeichnis")
    ap.add_argument("--embodiment-tag", default="unitree_g1", help="Embodiment-Schlüssel")
    ap.add_argument("--out", required=True, help="Ausgabe-JSON-Pfad")
    args = ap.parse_args()

    from transformers import AutoProcessor

    print(f"[dims] Lade Prozessor von {args.model_path} …", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    emb = args.embodiment_tag
    # Beide Prozessoren (N1.6 wie N1.7) haben state_action_processor und
    # get_modality_configs. Fehlt eins davon, ist der Checkpoint etwas anderes als
    # erwartet — dann lieber eine lesbare Meldung als ein AttributeError.
    if not hasattr(processor, "state_action_processor") or not hasattr(
        processor, "get_modality_configs"
    ):
        print(
            f"[dims] FEHLER: Der Prozessor aus {args.model_path} hat weder "
            "state_action_processor noch get_modality_configs — das ist kein "
            "GR00T-N1.6-/N1.7-Checkpoint.",
            file=sys.stderr,
        )
        return 1
    stats = processor.state_action_processor.statistics
    if emb not in stats:
        print(f"[dims] FEHLER: Embodiment '{emb}' nicht in Statistiken. "
              f"Verfügbar: {list(stats.keys())}", file=sys.stderr)
        return 1

    modality_configs = processor.get_modality_configs()[emb]

    def group_dim(modality: str, group: str) -> int:
        # Dimension = Länge einer beliebigen Stat-Liste (mean/std/min/max/…).
        group_stats = stats[emb][modality][group]
        any_list = next(iter(group_stats.values()))
        return len(any_list)

    state_keys = list(modality_configs["state"].modality_keys)
    action_keys = list(modality_configs["action"].modality_keys)

    out = {
        "embodiment": emb,
        "state": {g: group_dim("state", g) for g in state_keys},
        "action": {g: group_dim("action", g) for g in action_keys},
        "state_keys": state_keys,
        "action_keys": action_keys,
        "action_horizon": len(modality_configs["action"].delta_indices),
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[dims] Geschrieben: {args.out}", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
