# TL;DR: Menue-Parameter der Aktion clean (Container entfernen) von server_rl_run.sh.
action clean "Container entfernen (Daten bleiben)" \
  --rank 20 \
  --group "Werkzeuge" \
  --hint  "Entfernt den langlebigen Workbench-Container und den Web-Viewer. Das Host-Datenverzeichnis mit Checkpoints, Shader-Cache und Logs bleibt unangetastet."

param HF_TOKEN secret "" expert "HuggingFace-Token"
param CHECKPOINT_PATH path "/data/checkpoints/groot-g1dex3-checkpoint" expert \
  "Checkpoint im Container" "Nicht noetig."
param HF_CHECKPOINT_REPO str "luca-mue/groot-g1dex3-checkpoint" expert \
  "HuggingFace-Repo des Checkpoints" "Nicht noetig."
