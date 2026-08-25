# TL;DR: Menue-Parameter der Aktion train von kisski_submit.sh (BC-Fine-tuning).
action train "BC-Fine-tuning einreichen" \
  --rank 10 --group "Training" \
  --cli "Training/kisski_submit.sh" \
  --hint "Klont das Repo selbst und laedt Modell und Datensatz von HuggingFace — kein manuelles rsync. Tokens liest der Job aus ~/.hf_token und ~/.wandb_key."

group "Laufumfang"
param MAX_STEPS int 44000 basic \
  "Trainingsschritte" \
  "Der Mehr-GPU-Default auf KISSKI ist 44000, gegen 20000 im Container-Default." \
  --range 100:200000
param GLOBAL_BATCH_SIZE int 32 basic \
  "Globale Batch-Groesse" \
  "32 ist der Vorgabewert der Partition kisski (A100, 80 GB). Auf H100 sind 128 moeglich." \
  --range 1:512
param NUM_GPUS int 4 advanced "Anzahl GPUs" --range 1:8 \
  --default-from "#SBATCH --gpus in kisski_submit.sh"

group "Verfahren"
param TUNE_VISUAL bool 0 basic "Vision-Encoder mittrainieren"
param USE_COTRAIN bool 0 basic "Co-Training echt + gerendert (Schritt 4)"
when '[[ "${USE_COTRAIN:-0}" == 1 ]]'
param COTRAIN_HF_REPO str "" basic "HF-Repo des gerenderten Datensatzes" \
  --default-from "run_finetuning_cotrain.sh, leer = optional"
when '[[ "${USE_COTRAIN:-0}" == 1 ]]'
param COTRAIN_MIX_RATIO float 0.5 basic \
  "Anteil gerenderter Stichproben" \
  "Fuer den ersten Lauf 0.25 setzen: bei 60 gerenderten gegen 240 echte Episoden saehe das Modell sonst jedes Sim-Bild viermal so oft wie jedes echte." \
  --range 0:1

group "Validierung"
param TRAIN_TEST_SPLIT bool 0 basic "80/20-Split aktivieren" \
  "Ohne zurueckgehaltene Episoden kann checkpoint_sweep.py hinterher nichts messen."

group "Ablauf"
param SKIP_DOWNLOAD bool 0 advanced "HF-Download ueberspringen"
param SKIP_CONVERT  bool 0 advanced "Konvertierung ueberspringen"
param SKIP_GIT_PULL bool 0 advanced "Repo-Aktualisierung ueberspringen"
param RESUME bool 0 advanced \
  "Vorhandenen Lauf fortsetzen" \
  "Der richtige Weg nach einem Walltime-Abbruch. Ohne RESUME=1 bricht der Lauf ab, wenn schon Checkpoints im Namensraum liegen."
