# TL;DR: Menue-Parameter der Aktion openloop von kisski_open_loop_eval.sh (Checkpoint-Sweep).
action openloop "Checkpoint-Sweep (Open-Loop-Eval) einreichen" \
  --rank 10 --group "Auswerten" \
  --cli "Training/kisski_open_loop_eval.sh" \
  --hint "Misst MSE/MAE jedes Checkpoints auf den ZURUECKGEHALTENEN Episoden — braucht also einen Lauf mit TRAIN_TEST_SPLIT=1. Der Fork hat keine Validierung waehrend des Trainings, das hier ist der Ersatz."

param KISSKI_TIME str "04:00:00" basic "Walltime (HH:MM:SS)" \
  "Deutlich kuerzer als ein Trainingsjob — der Sweep ist reine Inferenz." \
  --override "Eval braucht keine 48 h; kuerzere Walltime kommt schneller durch die Warteschlange"
