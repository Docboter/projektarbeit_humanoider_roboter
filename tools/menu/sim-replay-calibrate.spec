# TL;DR: Menue-Parameter der Aktion replay-calibrate (Pixel-zu-Tisch-Homographien) von server_rl_run.sh.
action replay-calibrate "Replay: Pixel-zu-Tisch-Kalibrierung lernen" \
  --rank 41 \
  --group "Werkzeuge" \
  --needs "replay-prepare" \
  --hint  "Aus Bewegungsbeginn, stabilen Top-Face-Pixeln und wenigen direkt gesetzten FK-Zustaenden zwei Homographien lernen. Vollstaendige Episoden bleiben als Holdout getrennt."

param HF_TOKEN secret "" expert "HuggingFace-Token" "Nur noetig, wenn der Datensatz noch fehlt." \
  --default-from "Geheimnis - hat per Definition keinen Default"
param CHECKPOINT_PATH path "/data/checkpoints/groot-g1dex3-checkpoint" expert \
  "Checkpoint im Container" "Nicht noetig."
param HF_CHECKPOINT_REPO str "luca-mue/groot-g1dex3-checkpoint" expert \
  "HuggingFace-Repo des Checkpoints" "Nicht noetig."
param REPLAY_CALIBRATION_NUM_EPISODES int 40 basic "Episoden fuer die Kalibrierung" --range 2:500
param REPLAY_CALIBRATION_HOLDOUT_RATIO float 0.2 advanced "Holdout-Anteil"
param REPLAY_CALIBRATION_SEED int 17 expert "Seed der Holdout-Auswahl"
param REPLAY_DATASET path "/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset" expert \
  "Realdatensatz im Container"
param REPLAY_WORK path "/data/cube_replay/work" expert "Arbeitsverzeichnis"
