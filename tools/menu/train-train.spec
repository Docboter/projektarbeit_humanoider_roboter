action train "Neues Training starten" \
  --cli "" \
  --rank 10 --group "Training" \
  --hint  "Zieht das Image und startet den autonomen Container-Entrypoint: Download, v3->v2-Konvertierung, Training. Der Container behaelt alle Daten — kein --rm, kein Host-Mount." \
  --state 'docker ps -a --format "{{.Names}}" 2>/dev/null | grep -qx "${CONTAINER_NAME:-groot-train}" && echo "! Container existiert schon"'

group "Zugang"
param HF_TOKEN secret "" basic "HuggingFace-Token" \
  "Pflicht. Dauerhaft besser in .env.local ablegen (: \"\${HF_TOKEN:=hf_...}\") — dann fragt das Menue hier nicht mehr." \
  --default-from "Geheimnis - hat per Definition keinen Default"
# Erneut deklariert, damit der Key NACH dem Token gefragt wird: die Reihenfolge folgt
# der letzten Deklaration, und _common-train.spec nennt ihn vor dieser Datei.
param WANDB_API_KEY secret "" basic \
  "Weights-&-Biases-Key" \
  "Leer lassen laeuft ohne W&B. Bei einem Lauf ueber Stunden oder Tage ist das Dashboard aber der einzige bequeme Weg, den Verlauf zu sehen." \
  --default-from "Geheimnis - hat per Definition keinen Default"
group "Laufumfang"
param MAX_STEPS int 30000 basic \
  "Trainingsschritte" \
  "30000 ist der uebliche Wert auf einer Karte. Auf KISSKI mit mehreren GPUs eher 44000. Lauf 3 hat gezeigt, dass mehr nicht besser ist: der Checkpoint-Sweep ergab eine U-Kurve, der beste Checkpoint lag bei 30000 und der letzte war 25 % schlechter." \
  --range 100:200000

param GLOBAL_BATCH_SIZE int 8 basic \
  "Globale Batch-Groesse" \
  "Die erste Stellschraube bei OOM. Richtwerte: 24 GB VRAM -> 1-2 (sehr langsam), 32 GB -> 8, 80 GB (A100) -> 64, 94 GB (H100) -> 128. Ein volles Fine-tuning braucht laut NVIDIA mindestens 40 GB." \
  --range 1:512 \
  --suggest '_menu_suggest_batch_size'

param NUM_GPUS int 1 basic \
  "Anzahl GPUs" \
  "Mehr als 1 startet torchrun. Die globale Batch-Groesse teilt sich auf die Karten auf." \
  --range 1:8 \
  --suggest 'n=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l); [[ ${n:-0} -gt 0 ]] && echo "$n|$n GPU(s) erkannt"'

group "Verfahren"
note "Genau eines der drei Verfahren waehlen. USE_COTRAIN hat Vorrang vor TUNE_VISUAL (es setzt --tune_visual selbst)."

param TUNE_VISUAL bool 0 basic \
  "Vision-Encoder mittrainieren" \
  "Eigener Namensraum blockstacking_vision, LLM bleibt eingefroren, deutlich mehr VRAM. Lauf 34 hat den TUNE_VISUAL-Checkpoint bei 27,6 % Fingerspanne gemessen gegen 20,5 % ohne — 'lifted' blieb aber 0/10."

param USE_COTRAIN bool 0 basic \
  "Co-Training auf echten UND gerenderten Bildern" \
  "Schritt 4. Namensraum blockstacking_cotrain, setzt --tune_visual selbst und hat Vorrang vor TUNE_VISUAL. Braucht einen gerenderten Datensatz aus 'server_rl_run.sh render'. Details: docs/training/co-training.md"

when '[[ "${USE_COTRAIN:-0}" == 1 ]]'
param COTRAIN_MIX_RATIO float 0.5 basic \
  "Anteil gerenderter Bilder" \
  "0.25 ist der empfohlene Wert. Zu hoch, und das Modell lernt die Sim statt der Aufgabe." \
  --range 0:1

when '[[ "${USE_COTRAIN:-0}" == 1 ]]'
param COTRAIN_HF_REPO str "" basic \
  "HF-Repo des gerenderten Datensatzes" \
  "Alternativ COTRAIN_DATASET_PATH, wenn der Datensatz schon im Container liegt." \
  --default-from "run_finetuning_cotrain.sh, leer = optional"
when '[[ "${USE_COTRAIN:-0}" == 1 ]]'
param COTRAIN_DATASET_PATH path "/data/cotrain/g1_dex3_rendered" advanced "Pfad des gerenderten Datensatzes im Container"

group "Validierung"
param TRAIN_TEST_SPLIT bool 0 basic \
  "80/20-Split aktivieren" \
  "Haelt Test-Episoden zurueck, damit checkpoint_sweep.py hinterher offene-Schleife-MSE auf UNGESEHENEN Daten messen kann. Der Fork hat keine Validierung waehrend des Trainings (enable_open_loop_eval ist tote Konfiguration), deshalb ist das die einzige Stelle, an der man sie bekommt. Ohne Split ist jede spaetere Checkpoint-Auswahl geraten."
when '[[ "${TRAIN_TEST_SPLIT:-0}" == 1 ]]'
param TRAIN_SPLIT_RATIO float 0.8 advanced "Anteil Trainingsepisoden" --range 0.1:0.99

group "Augmentierung"
param USE_AUGMENTATION bool 1 advanced \
  "Bildaugmentierung / Domain-Randomisierung" \
  "Color-Jitter und Rotation. Standardmaessig an; 0 schaltet sie ausdruecklich ab."

group "Ablauf"
param SKIP_DOWNLOAD bool 0 advanced "HF-Download ueberspringen" "Wenn Modell und Datensatz schon im Container liegen."
param SKIP_CONVERT  bool 0 advanced "v3->v2-Konvertierung ueberspringen" "Wenn modality.json bereits existiert."
param SKIP_TRAIN    bool 0 advanced "Nur einrichten, dann Shell" "Zum Nachsehen, ob Download und Konvertierung sauber durchgelaufen sind."
param SHELL_ON_ERROR bool 0 advanced \
  "Bei Fehler in eine Shell fallen" \
  "Empfohlen beim ersten Lauf auf einer neuen Maschine: der Container bleibt stehen, statt den Zustand mitzunehmen."

group "Protokollierung"
param WANDB_PROJECT str "gr00t-g1-dex3" advanced "W&B-Projektname"
param WANDB_MODE choice offline advanced \
  "W&B-Betriebsart" \
  "'offline' schreibt nur lokal — der richtige Wert auf einem Rechenknoten ohne Netz (KISSKI). Nachtraeglich synchronisieren: docs/training/wandb-offline-sync.md" \
  --options "online:direkt melden;offline:nur lokal schreiben;disabled:ganz aus"
