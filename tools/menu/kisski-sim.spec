# TL;DR: Menue-Parameter der Aktion sim von kisski_sim_submit.sh (Sim-Eval, jupyter-Partition).
action sim "Sim-Eval einreichen" \
  --rank 10 --group "Simulation" \
  --cli "Simulation/kisski_sim_submit.sh" \
  --hint "Laeuft auf der jupyter-Partition (RTX 5000). Isaac Sim braucht RT-Cores; die Rechenpartitionen haben keine."

param KISSKI_PARTITION choice jupyter basic "Partition" \
  --options "jupyter:RTX 5000 — die einzige mit RT-Cores;kisski:A100 (kein Rendering);kisski-h100:H100 (kein Rendering)" \
  --override "Sim-Eval braucht RT-Cores; nur jupyter hat sie" \
  --default-from "#SBATCH --partition in kisski_sim_submit.sh"
param NUM_EPISODES int 20 basic "Anzahl Eval-Episoden" --range 1:200
param EPISODE_LENGTH_S int 0 basic "Zeitbudget je Episode (Sekunden)" --range 0:3600
