action rl "Echter RL-Lauf (FPO, Vordergrund)" \
  --rank 10 \
  --group "Trainieren" \
  --needs "setup, check" \
  --hint  "Verfeinert den BC-Checkpoint per FPO in der Block-Stacking-Sim. Laeuft je nach RL_ITERATIONS sehr lange und blockiert das Terminal — in tmux starten. Checkpoints landen unter <datenverzeichnis>/g1_dex3_rl/."

group "Zugang"
param HF_TOKEN secret "" basic "HuggingFace-Token" \
  --default-from "Geheimnis - hat per Definition keinen Default"
param WANDB_API_KEY secret "" basic \
  "Weights-&-Biases-Key" \
  "Leer lassen laeuft ohne W&B. Bei einem tagelangen Lauf ist das Dashboard aber der einzige bequeme Weg, den Fortschritt zu verfolgen." \
  --default-from "Geheimnis - hat per Definition keinen Default"
group "Laufumfang"
param RL_NUM_ENVS int 16 basic \
  "Parallele Sim-Envs" \
  "Die wichtigste Speicher-Stellschraube. Bei knappem VRAM zuerst hier herunter (4 ist ein brauchbarer Start auf einer Karte)." \
  --range 1:256 --default-from "Simulation/scripts/entrypoint_rl.sh"
param RL_ITERATIONS int 500 basic \
  "RL-Iterationen (Rollout + Update)" \
  --range 1:100000 --default-from "Simulation/scripts/entrypoint_rl.sh"
param RL_ROLLOUT_STEPS int 32 advanced "Env-Steps je Rollout und Env" \
  --default-from "Simulation/scripts/entrypoint_rl.sh"
param RL_SAVE_EVERY int 100 advanced "Checkpoint alle N Iterationen" \
  --default-from "Simulation/scripts/entrypoint_rl.sh"

group "Optimierung"
param RL_LR str "1e-5" advanced "Lernrate" \
  --default-from "Simulation/scripts/entrypoint_rl.sh"
param RL_KL_COEF float 0.1 expert "KL-Koeffizient" --default-from "Simulation/scripts/entrypoint_rl.sh"
param RL_CLIP float 0.2 expert "Clipping" --default-from "Simulation/scripts/entrypoint_rl.sh"
param RL_MINIBATCH_SIZE int 0 expert "Minibatch-Groesse (Speicher)" --default-from "Simulation/scripts/entrypoint_rl.sh"
param RL_EPOCHS_PER_ITER int 0 expert "Epochen je Iteration (Speicher)" --default-from "Simulation/scripts/entrypoint_rl.sh"
param RL_FPO_MC_SAMPLES int 0 expert "Monte-Carlo-Samples fuer FPO (Speicher)" --default-from "Simulation/scripts/entrypoint_rl.sh"

group "Hardware"
param RL_GPUS str '"device=1,0"' advanced \
  "GPU-Auswahl" \
  "Die ZUERST genannte Karte wird im Container zu cuda:0 und traegt Rendering, Policy und Optimizer; die zweite bekommt nur das eingefrorene Referenzmodell. Vor einem langen Lauf 'nvidia-smi' pruefen. Einzelkarte: \"device=0\" — das Referenzmodell rueckt dann mit auf."
param RL_REF_DEVICE choice auto advanced \
  "Geraet des Referenzmodells" \
  --options "auto:selbst entscheiden;same:auf dieselbe Karte;cuda:1:feste Karte"

group "Live-Ansicht"
param LIVE_VIEW bool 0 basic \
  "MJPEG-Bilder im Browser" \
  "Fuer den tagelangen RL-Lauf der richtige Weg: zustandslos, beliebig viele Zuschauer, per 'ssh -L' tunnelbar, kostet keinen extra Renderdurchgang."
param RL_WANDB_VIDEO_EVERY int 0 advanced \
  "Rollout-Video alle N Iterationen ins W&B" \
  "0 = aus." \
  --default-from "rl_finetune.py --wandb-video-every"
