# TL;DR: Menue-Parameter der Aktion yawcheck (Gierwinkel-Schaetzer abnehmen) von server_rl_run.sh.
action yawcheck "Gierwinkel-Schaetzer gegen synthetische Wuerfel abnehmen" \
  --rank 15 \
  --group "Co-Training vorbereiten" \
  --hint  "Braucht weder Datensatz noch Isaac noch GPU, laeuft in Sekunden. Rendert Wuerfel BEKANNTER Drehung durch dasselbe Kameramodell und misst zurueck — der einzige Weg, bei dem der wahre Winkel wirklich bekannt ist. Nach jeder Aenderung am Schaetzer zuerst das hier."

param HF_TOKEN secret "" expert "HuggingFace-Token" "Nicht noetig."
param CHECKPOINT_PATH path "/data/checkpoints/groot-g1dex3-checkpoint" expert \
  "Checkpoint im Container" "Nicht noetig — der Schaetzer ist reine Bildverarbeitung."
param HF_CHECKPOINT_REPO str "luca-mue/groot-g1dex3-checkpoint" expert \
  "HuggingFace-Repo des Checkpoints" "Nicht noetig."

group "Tor"
param LAYOUT_YAW_TOLERANCE float 8.0 advanced \
  "Erlaubte Uneinigkeit beider Kameras (Grad)" \
  "Das einzige wirksame Tor. Zwei unabhaengig aufgestellte Kameras stimmen bei Rauschen nicht ueberein — deshalb traegt es allein." \
  --default-from "extract_block_layout.py --yaw-tolerance"
param LAYOUT_MIN_SQUARENESS float 1.0 expert \
  "Verlangte Diagonale/Kante der Deckflaeche" \
  "1,0 = aus. Auf echten Frames trennt die Formprobe nichts (Messung 2026-08-25), sie kostete nur 8 von 9 brauchbaren Wuerfeln." \
  --default-from "extract_block_layout.py --min-squareness"
