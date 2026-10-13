# TL;DR: Menue-Parameter der Aktion topface (Deckflaechenschnitt vermessen) von server_rl_run.sh.
action topface "Deckflaechenschnitt auf echten Frames vermessen" \
  --rank 26 \
  --group "Co-Training vorbereiten" \
  --hint  "Die Wuerfeloberseite IST ein 5-cm-Quadrat — welche Schwellenregel ihr am naechsten kommt, ist damit eine Messung und keine Meinung. Vergleicht die gemessene Schwelle (Otsu) gegen feste Perzentile an Breite (Soll 5,0 cm) und Formprobe (Soll 1,41) und druckt eine Eichtabelle fuer die Formprobe. Braucht keine GPU."

param HF_TOKEN secret "" expert "HuggingFace-Token" "Nicht noetig."
param CHECKPOINT_PATH path "/data/checkpoints/groot-g1dex3-checkpoint" expert \
  "Checkpoint im Container" "Nicht noetig."
param HF_CHECKPOINT_REPO str "luca-mue/groot-g1dex3-checkpoint" expert \
  "HuggingFace-Repo des Checkpoints" "Nicht noetig."

group "Umfang"
param TOPFACE_EPISODES int 5 basic "Anzahl Episoden" \
  "Fuenf reichen: gemessen werden 3 Wuerfel x 2 Kameras je Episode." --range 1:60
param LAYOUT_EPISODE_IDS str "" advanced \
  "Nur diese Episoden" \
  "Durch Leerzeichen getrennt, z. B. '0 1 2 8 12'. Leer = fortlaufend ab 0." \
  --default-from "extract_block_layout.py --episode-ids (None)"
param TOPFACE_RULES str "" advanced \
  "Nur diese Regeln vergleichen" \
  "Z. B. 'otsu q50 q70'. Leer = alle." \
  --default-from "extract_block_layout.py --rules (alle)"
group "Tor"
param LAYOUT_YAW_TOLERANCE float 8.0 expert \
  "Erlaubte Uneinigkeit beider Kameras (Grad)" \
  --default-from "extract_block_layout.py --yaw-tolerance"
param LAYOUT_MIN_SQUARENESS float 1.0 expert \
  "Verlangte Diagonale/Kante der Deckflaeche" "1,0 = aus." \
  --default-from "extract_block_layout.py --min-squareness"
