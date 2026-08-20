# TL;DR: Gemeinsame KISSKI-Parameter (Partition, Walltime, Zugang) fuer kisski_menu.sh.
# _common-kisski.spec — was alle KISSKI-Jobs teilen.

group "Rechenzeit"
param KISSKI_PARTITION choice kisski basic \
  "Partition" \
  "kisski hat A100 mit 80 GB, kisski-h100 hat H100 mit 94 GB. Beide sind auf 48 h Walltime begrenzt. Die H100-Partition ist meist staerker belegt — bei knapper Zeit lieber A100 mit kleinerer Batch-Groesse." \
  --options "kisski:A100, 80 GB;kisski-h100:H100, 94 GB;jupyter:RTX 5000 (nur Sim-Eval)" \
  --default-from "#SBATCH --partition in den kisski_*.sh"

param KISSKI_TIME str "48:00:00" basic \
  "Walltime (HH:MM:SS)" \
  "Maximum sind 48 Stunden. Laenger geht nicht — ein Lauf, der mehr braucht, muss ueber RESUME=1 fortgesetzt werden." \
  --default-from "#SBATCH --time in den kisski_*.sh"

group "Ablage"
param KISSKI_PROJECT_DIR path "/mnt/vast-kisski/projects/kisski-humrob" advanced \
  "Projektverzeichnis auf der VAST-Ablage" \
  "Daraus leiten die Skripte SIF, Datenverzeichnis und Repo-Klon ab. Anders als beim Docker-Weg bleiben die Daten hier ueber Jobs hinweg erhalten."

group "Zugang"
# Auf KISSKI liest der Job die Tokens aus ~/.hf_token und ~/.wandb_key (mode 600) —
# ueber die Umgebung kaemen sie wegen SBATCH_EXPORT=none ohnehin nicht an. Deshalb hier
# heruntergestuft: danach zu fragen waere irrefuehrend.
param HF_TOKEN secret "" expert "HuggingFace-Token" \
  "Auf KISSKI aus ~/.hf_token, nicht aus der Umgebung." \
  --default-from "Geheimnis - hat per Definition keinen Default"
param WANDB_API_KEY secret "" expert "Weights-&-Biases-Key" \
  "Auf KISSKI aus ~/.wandb_key." \
  --default-from "Geheimnis - hat per Definition keinen Default"
