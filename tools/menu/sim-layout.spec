# TL;DR: Menue-Parameter der Aktion layout (Wuerfellage aus Realbildern) von server_rl_run.sh.
action layout "Wuerfellage aus den Realbildern lesen" \
  --rank 20 \
  --group "Co-Training vorbereiten" \
  --needs "layoutcheck (Kameramodell pruefen)" \
  --hint  "Farbblob (rot/gruen/gelb) im ersten Frame, Strahl durch den Schwerpunkt auf die Wuerfelebene. Braucht keine GPU, Minuten statt Stunden. Das ist die RICHTIGE Quelle fuer 'render' — der Greifpunkt aus scan.json ist das Minimum der Fingeroeffnung ueber die ganze Episode und liegt bei knapp der Haelfte der Griffe auf dem Transportweg statt am Pick." \
  --state '[[ -f "${HOST_DATA_DIR:-$HOME/groot-rl-data}/cotrain/layout.json" ]] && echo "✓ layout.json liegt vor"'

param HF_TOKEN secret "" advanced "HuggingFace-Token" "Nur noetig, wenn der Datensatz noch fehlt." \
  --default-from "Geheimnis - hat per Definition keinen Default"
param CHECKPOINT_PATH path "/data/checkpoints/groot-g1dex3-checkpoint" expert \
  "Checkpoint im Container" "Nicht noetig — die Wuerfellage kommt aus den Realbildern, ohne Modell."
param HF_CHECKPOINT_REPO str "luca-mue/groot-g1dex3-checkpoint" expert \
  "HuggingFace-Repo des Checkpoints" "Nicht noetig."
group "Umfang"
param RENDER_EPISODES int 60 basic "Anzahl Episoden" --range 1:500
param LAYOUT_OUT path "/data/cotrain/layout.json" advanced "Ausgabedatei im Container"
param LAYOUT_OVERWRITE bool 0 advanced "Vorhandene layout.json ueberschreiben"
param LAYOUT_DEBUG_DIR path "/data/cotrain/layout_debug" advanced \
  "Debug-Bilder ablegen" \
  "Leer = keine. Sonst landen die Blob-Masken dort — der schnellste Weg zu sehen, warum ein Wuerfel nicht gefunden wurde."
param LAYOUT_BIAS str "" expert "Korrektur 'dx dy' in Metern" \
  --default-from "extract_block_layout.py --bias"
