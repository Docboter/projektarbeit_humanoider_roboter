# TL;DR: Menue-Parameter der Aktion replay-prepare (Replay-Quelldatensatz pruefen) von server_rl_run.sh.
action replay-prepare "Replay: Realdatensatz holen und Schema pruefen" \
  --rank 40 \
  --group "Werkzeuge" \
  --hint  "Schritt 1 der physikbasierten Replay-Videos (aelterer Weg; 'render' nutzt das Bild-Layout). Prueft Schema, vier Kameras, 30 Hz und 28-DoF-State/Actions, laedt keine Modellgewichte."

param HF_TOKEN secret "" advanced "HuggingFace-Token" "Nur noetig, wenn der Datensatz noch fehlt." \
  --default-from "Geheimnis - hat per Definition keinen Default"
param CHECKPOINT_PATH path "/data/checkpoints/groot-g1dex3-checkpoint" expert \
  "Checkpoint im Container" "Nicht noetig - Replay spielt Realaktionen ab, ohne Modell."
param HF_CHECKPOINT_REPO str "luca-mue/groot-g1dex3-checkpoint" expert \
  "HuggingFace-Repo des Checkpoints" "Nicht noetig."
param REPLAY_DATASET path "/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset" advanced \
  "Realdatensatz im Container"
param REPLAY_NUM_EPISODES int 10 basic "Anzahl Episoden" --range 1:500
param REPLAY_START_EPISODE int 0 advanced "Erste Episode"
param REPLAY_EPISODE_IDS str "" advanced "Nur diese Episoden (Leerzeichen-getrennt)" \
  --default-from "leer = fortlaufend ab REPLAY_START_EPISODE"
