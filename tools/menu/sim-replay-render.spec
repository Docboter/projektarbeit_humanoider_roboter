# TL;DR: Menue-Parameter der Aktion replay-render (Realaktionen in der Sim abspielen) von server_rl_run.sh.
action replay-render "Replay: Realaktionen abspielen (Videos/Datensatz)" \
  --rank 43 \
  --group "Werkzeuge" \
  --needs "replay-poses" \
  --hint  "Originale 28-DoF-Actions bei exakt 30 Hz. Jeder Wuerfel wird einmal vor Frame 0 gesetzt und danach nur von PhysX bewegt. Videos (Standard) oder LeRobot-v2.1-Datensatz."

param HF_TOKEN secret "" expert "HuggingFace-Token" "Nur noetig, wenn der Datensatz noch fehlt." \
  --default-from "Geheimnis - hat per Definition keinen Default"
param CHECKPOINT_PATH path "/data/checkpoints/groot-g1dex3-checkpoint" expert \
  "Checkpoint im Container" "Nicht noetig."
param HF_CHECKPOINT_REPO str "luca-mue/groot-g1dex3-checkpoint" expert \
  "HuggingFace-Repo des Checkpoints" "Nicht noetig."
param REPLAY_OUTPUT_MODE choice videos basic "Ausgabe" \
  --options "videos:fuenf MP4s je Episode;dataset:trainierbarer LeRobot-v2.1-Datensatz"
param REPLAY_NUM_EPISODES int 10 basic "Anzahl Episoden" --range 1:500
param REPLAY_START_EPISODE int 0 advanced "Erste Episode"
param REPLAY_EPISODE_IDS str "" advanced "Nur diese Episoden (Leerzeichen-getrennt)" \
  --default-from "leer = fortlaufend ab REPLAY_START_EPISODE"
param REPLAY_MAX_FRAMES int 0 advanced "Frames je Episode begrenzen (0 = alle)"
param REPLAY_OVERWRITE bool 0 advanced "Vorhandene Ausgaben ueberschreiben"
param DR_ENABLED bool 0 advanced "Domain-Randomisierung"
param REPLAY_OUT path "/data/cube_replay/videos" expert "Video-Ausgabe"
param REPLAY_DATASET_OUT path "/data/cube_replay/dataset" expert "Datensatz-Ausgabe"
