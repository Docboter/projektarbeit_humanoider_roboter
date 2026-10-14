# TL;DR: Menue-Parameter der Aktion oracle (Orakel-Replay, obere Schranke der Eval) von server_rl_run.sh.
action oracle "Orakel-Replay: Demo-Aktionen statt Policy (obere Schranke)" \
  --rank 31 \
  --group "Messen" \
  --needs "layout mit LAYOUT_HELD_OUT=1" \
  --hint  "Die Eval ohne Modell: je Episode spielt der Datensatz seine aufgezeichneten Aktionen bei 30 Hz ab, auf derselben Wuerfellage, mit denselben Meilensteinen wie 'eval'. Greift schon die Demo nicht, ist die Sim der Blocker und jedes Policy-Training geht ins Leere (Review 10-13 §2.3 Nr. 1). Misst zusaetzlich den Tracking-Fehler Sim<->Aufzeichnung." \
  --state '[[ -f "${HOST_DATA_DIR:-$HOME/groot-rl-data}/eval/layout_heldout.json" ]] && echo "✓ layout_heldout.json liegt vor" || echo "! braucht vorher LAYOUT_HELD_OUT=1 layout"'

param HF_TOKEN secret "" expert "HuggingFace-Token" "Nur noetig, wenn der Datensatz noch fehlt." \
  --default-from "Geheimnis - hat per Definition keinen Default"
param CHECKPOINT_PATH path "/data/checkpoints/groot-g1dex3-checkpoint" expert \
  "Checkpoint im Container" "Nicht noetig — es laeuft kein Modell. Liefert nur das USD-Asset, falls es dort liegt."
param HF_CHECKPOINT_REPO str "luca-mue/groot-g1dex3-checkpoint" expert \
  "HuggingFace-Repo des Checkpoints" "Nicht noetig."

group "Laufumfang"
param NUM_EPISODES int 20 basic \
  "Anzahl Episoden" \
  "Dieselbe Zahl wie in 'eval', dann sind es dieselben Layout-Episoden (gepaarter Vergleich)." \
  --range 1:200
param EPISODE_LENGTH_S int 120 basic \
  "Zeitbudget je Episode (Sekunden)" \
  "Die Test-Episoden dauern 22-37 s; danach haelt die Hand die letzte Aktion. 120 wie in 'eval'." \
  --range 1:3600 \
  --override "do_oracle setzt 120 statt der 300 s des cfg-Defaults, den 'eval' bei 0 nimmt"
param EVAL_LAYOUT path "/data/eval/layout_heldout.json" basic \
  "Wuerfel-Startlagen (layout.json)" \
  "Bestimmt zugleich, WELCHE Episode abgespielt wird: Eintrag n = Datensatz-Episode n."
param ORACLE_DATASET path "/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset" advanced \
  "Datensatz mit den Aktionen (v2.1)"
param EXECUTION_HORIZON int 8 advanced \
  "Aktionen je Chunk" \
  "Wie 'eval'. Physikalisch ohne Wirkung (die Aktionen sind fest); bestimmt nur, wie oft der Tracking-Fehler abgetastet wird." \
  --range 1:16
group "Messaufbau"
param DR_ENABLED bool 0 advanced "Domain-Randomisierung" "Fuer die Schranke aus."
param DR_SEED int 0 advanced "Seed"
param SCENE_CAM bool 1 advanced "Uebersichtskamera fuer die Videos rendern"
param ORACLE_OUT path "/data/sim_results/oracle.json" expert "Ergebnisdatei im Container"
param ORACLE_VIDEO_DIR path "/data/sim_videos/oracle" expert "Video-Verzeichnis im Container"
