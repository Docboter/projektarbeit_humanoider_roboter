# TL;DR: Menue-Parameter der Aktion rl von kisski_rl_submit.sh (RL-Fine-tuning FPO).
action rl "RL-Fine-tuning (FPO) einreichen" \
  --rank 20 --group "Training" \
  --cli "Training/kisski_rl_submit.sh" \
  --hint "ACHTUNG: braucht eine GPU mit RT-Cores fuer das Kamera-Rendering. Die KISSKI-A100 und -H100 haben keine. Das Skript ist eine Vorlage und prueft das selbst."

param RL_NUM_ENVS int 16 basic "Parallele Sim-Envs" \
  --default-from "Simulation/scripts/entrypoint_rl.sh"
param RL_ITERATIONS int 500 basic "RL-Iterationen" \
  --default-from "Simulation/scripts/entrypoint_rl.sh"
