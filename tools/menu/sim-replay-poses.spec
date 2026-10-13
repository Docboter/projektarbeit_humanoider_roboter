# TL;DR: Menue-Parameter der Aktion replay-poses (Wuerfel-Anfangsposen fuer Replay) von server_rl_run.sh.
action replay-poses "Replay: Wuerfel-Anfangsposen bestimmen" \
  --rank 42 \
  --group "Werkzeuge" \
  --needs "replay-calibrate" \
  --hint  "Einmalige Anfangspose aller Wuerfel aus stabilen Realframes und den geprueften Homographien. Kein Greifpunkt-, Zufalls- oder Alt-Layout-Fallback."

param HF_TOKEN secret "" expert "HuggingFace-Token" "Nur noetig, wenn der Datensatz noch fehlt." \
  --default-from "Geheimnis - hat per Definition keinen Default"
param CHECKPOINT_PATH path "/data/checkpoints/groot-g1dex3-checkpoint" expert \
  "Checkpoint im Container" "Nicht noetig."
param HF_CHECKPOINT_REPO str "luca-mue/groot-g1dex3-checkpoint" expert \
  "HuggingFace-Repo des Checkpoints" "Nicht noetig."
param REPLAY_NUM_EPISODES int 10 basic "Anzahl Episoden" --range 1:500
param REPLAY_START_EPISODE int 0 advanced "Erste Episode"
param REPLAY_EPISODE_IDS str "" advanced "Nur diese Episoden (Leerzeichen-getrennt)" \
  --default-from "leer = fortlaufend ab REPLAY_START_EPISODE"
param REPLAY_OVERWRITE bool 0 advanced "Vorhandene Posen neu bestimmen"
param REPLAY_WORK path "/data/cube_replay/work" expert "Arbeitsverzeichnis"
