# TL;DR: Menue-Parameter der Aktion openloop von kisski_open_loop_eval.sh (Checkpoint-Sweep).
action openloop "Checkpoint-Sweep (Open-Loop-Eval) einreichen" \
  --rank 10 --group "Auswerten" \
  --cli "Training/kisski_open_loop_eval.sh" \
  --hint "Misst MSE/MAE jedes Checkpoints auf den ZURUECKGEHALTENEN Episoden — braucht also einen Lauf mit TRAIN_TEST_SPLIT=1. Der Fork hat keine Validierung waehrend des Trainings, das hier ist der Ersatz."

param KISSKI_TIME str "04:00:00" basic "Walltime (HH:MM:SS)" \
  "Deutlich kuerzer als ein Trainingsjob — der Sweep ist reine Inferenz." \
  --override "Eval braucht keine 48 h; kuerzere Walltime kommt schneller durch die Warteschlange"

group "GR00T-Generation"
param GROOT_VERSION choice 1.6 advanced \
  "GR00T-Generation" \
  "Muss zu den Checkpoints passen, die gesweept werden — N1.6 und N1.7 sind nicht wechselseitig ladbar. 1.7 braucht Modell und gated Backbone vorab im HF-Cache des Clusters." \
  --options "1.6:Python 3.10, Eagle-Backbone (Default);1.7:Python 3.12, Cosmos-Reason2-2B (gated, ungetestet)"
