# TL;DR: Menue-Parameter der Aktion sim von kisski_sim_submit.sh (Sim-Eval, jupyter-Partition).
action sim "Sim-Eval einreichen" \
  --rank 10 --group "Nicht auf diesem Cluster" \
  --cli "Simulation/kisski_sim_submit.sh" \
  --blocked "RTX 5000 zu alt -> IKR-Server" \
  --hint "UEBERHOLT. Der Job zielt auf die jupyter-Partition, weil nur deren Quadro RTX 5000 ueberhaupt RT-Cores hat. Sie ist aber Turing und damit eine Generation zu alt fuer Isaac Sim 4.x (docs/fehlerbehebung.md). Die Anleitung dazu liegt schon im Archiv. Sim-Eval laeuft auf dem IKR-Server: ./run.sh sim eval. Das Skript bleibt als Beleg im Repo."

param KISSKI_PARTITION choice jupyter basic "Partition" \
  --options "jupyter:RTX 5000 — die einzige mit RT-Cores;kisski:A100 (kein Rendering);kisski-h100:H100 (kein Rendering)" \
  --override "Sim-Eval braucht RT-Cores; nur jupyter hat sie" \
  --default-from "#SBATCH --partition in kisski_sim_submit.sh"
param NUM_EPISODES int 20 basic "Anzahl Eval-Episoden" --range 1:200
param EPISODE_LENGTH_S int 0 basic "Zeitbudget je Episode (Sekunden)" --range 0:3600
