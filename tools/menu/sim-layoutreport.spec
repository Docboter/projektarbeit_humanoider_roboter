# TL;DR: Menue-Parameter der Aktion layoutreport (Diagnose einer layout.json) von server_rl_run.sh.
action layoutreport "Ablesen, woran der Gierwinkel scheitert" \
  --rank 25 \
  --group "Co-Training vorbereiten" \
  --needs "layout" \
  --hint  "Liest eine fertige layout.json und sagt, WELCHES Tor zuschlaegt: Formprobe zu klein heisst Farbmaske oder Deckflaechenschnitt, Kameras uneins heisst der Winkel selbst ist verrauscht. Braucht keinen neuen Lauf und keine GPU. Zeigt ausserdem die Verteilung der durchgelassenen Wuerfel neben der aller Einzelmessungen — decken sie sich, formt das Tor nichts." \
  --state '[[ -f "${HOST_DATA_DIR:-$HOME/groot-rl-data}/cotrain/layout.json" ]] && echo "✓ layout.json liegt vor"'

param HF_TOKEN secret "" expert "HuggingFace-Token" "Nicht noetig."
param CHECKPOINT_PATH path "/data/checkpoints/groot-g1dex3-checkpoint" expert \
  "Checkpoint im Container" "Nicht noetig."
param HF_CHECKPOINT_REPO str "luca-mue/groot-g1dex3-checkpoint" expert \
  "HuggingFace-Repo des Checkpoints" "Nicht noetig."

group "Eingaben"
param LAYOUT_OUT path "/data/cotrain/layout.json" basic "Zu pruefende layout.json"
group "Tor"
param LAYOUT_YAW_TOLERANCE float 8.0 advanced \
  "Erlaubte Uneinigkeit beider Kameras (Grad)" \
  --default-from "extract_block_layout.py --yaw-tolerance"
param LAYOUT_MIN_SQUARENESS float 1.0 expert \
  "Verlangte Diagonale/Kante der Deckflaeche" "1,0 = aus." \
  --default-from "extract_block_layout.py --min-squareness"
