# TL;DR: Menue-Parameter der Aktion rl von kisski_rl_submit.sh (RL-Fine-tuning FPO).
action rl "RL-Fine-tuning (FPO) einreichen" \
  --rank 20 --group "Nicht auf diesem Cluster" \
  --cli "Training/kisski_rl_submit.sh" \
  --blocked "Vorlage, keine RT-Core-Partition" \
  --hint "VORLAGE, nicht einreichbar. RL rendert 4 Kameras und braucht dafuer RT-Cores; kisski (A100) und kisski-h100 haben keine, und eine RT-Core-Partition gibt es auf dem Cluster nicht. Das Skript traegt deshalb PLACEHOLDER_RTCORE_PARTITION und einen Guard, der auf A100/H100 abbricht. RL laeuft auf dem IKR-Server: ./run.sh sim rl. Kommt einmal eine passende Partition dazu, sind hier nur die zwei Platzhalter zu fuellen."

param RL_NUM_ENVS int 16 basic "Parallele Sim-Envs" \
  --default-from "Simulation/scripts/entrypoint_rl.sh"
param RL_ITERATIONS int 500 basic "RL-Iterationen" \
  --default-from "Simulation/scripts/entrypoint_rl.sh"
