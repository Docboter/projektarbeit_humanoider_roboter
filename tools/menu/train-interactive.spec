# TL;DR: Trainings-Launcher-Aktion --interactive: Shell statt Training starten.
action interactive "Shell statt Training" \
  --cli "--interactive" \
  --rank 40 --group "Werkzeuge" \
  --hint  "Startet den Container mit einer Shell, ohne den Entrypoint. Zum Nachsehen, Nachrechnen und fuer 'bash /scripts/run_finetuning.sh' von Hand."

param HF_TOKEN      secret "" expert "HuggingFace-Token"
param WANDB_API_KEY secret "" expert "Weights-&-Biases-Key"
