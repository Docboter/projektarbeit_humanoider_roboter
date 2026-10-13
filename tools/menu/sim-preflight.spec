# TL;DR: Menue-Parameter der Aktion preflight (Image/GPU-Check) von server_rl_run.sh.
action preflight "Image-Frische + GPU/RT-Cores pruefen" \
  --rank 10 \
  --group "Vorbereiten" \
  --hint  "Weist verbindlich nach, dass gr00t und flash-attn im Isaac-Sim-Python liegen und die Karte RT-Cores hat. Braucht kein HF_TOKEN. Der erste Lauf auf einer neuen Maschine gehoert hierhin." \
  --state 'ls -t "${HOST_DATA_DIR:-$HOME/groot-rl-data}"/logs/preflight-*.log 2>/dev/null | head -1 | grep -q . && echo "✓ schon gelaufen"'

param HF_TOKEN secret "" expert "HuggingFace-Token" "Fuer preflight nicht noetig."
param CHECKPOINT_PATH path "/data/checkpoints/groot-g1dex3-checkpoint" expert \
  "Checkpoint im Container" "Fuer preflight nicht noetig — die Aktion prueft Image und GPU, kein Modell."
param HF_CHECKPOINT_REPO str "luca-mue/groot-g1dex3-checkpoint" expert \
  "HuggingFace-Repo des Checkpoints" "Fuer preflight nicht noetig — die Aktion prueft Image und GPU, kein Modell."
