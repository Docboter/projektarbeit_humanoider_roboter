# TL;DR: Menue-Parameter der Aktion shell (interaktive Container-Shell) von server_rl_run.sh.
action shell "Interaktive Shell im Container" \
  --rank 10 \
  --group "Werkzeuge" \
  --hint  "Wird als einzige Aktion NICHT ins Log gespiegelt — interaktives -it vertraegt die tee-Pipe nicht."

param HF_TOKEN secret "" expert "HuggingFace-Token"
param CHECKPOINT_PATH path "/data/checkpoints/groot-g1dex3-checkpoint" expert \
  "Checkpoint im Container" "Nicht noetig."
param HF_CHECKPOINT_REPO str "luca-mue/groot-g1dex3-checkpoint" expert \
  "HuggingFace-Repo des Checkpoints" "Nicht noetig."
