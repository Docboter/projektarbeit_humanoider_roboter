# TL;DR: Menue-Parameter der Aktion eval (Closed-Loop-BC-Eval) von server_rl_run.sh.
action eval "BC-Erfolgsrate in der Sim (Closed Loop)" \
  --rank 30 \
  --group "Messen" \
  --needs "setup" \
  --hint  "Schritt 3 der Diagnosekette und der Nullpunkt jedes RL-Vergleichs. GR00T-Server plus Isaac-Lab-Client, geschlossene Schleife." \
  --state '[[ -d "${HOST_DATA_DIR:-$HOME/groot-rl-data}/checkpoints" ]] || echo "! braucht vorher setup"'

group "Zugang"
param HF_TOKEN secret "" basic "HuggingFace-Token" \
  "Wird fuer den Checkpoint-Download gebraucht. Dauerhaft besser in .env.local ablegen — dann fragt das Menue hier nicht mehr." \
  --default-from "Geheimnis - hat per Definition keinen Default"
group "Laufumfang"
param NUM_EPISODES int 20 basic \
  "Anzahl Eval-Episoden" \
  "20 ist der Standard fuer vergleichbare Zahlen; 2 fuer einen Rauchtest." \
  --range 1:200

param EPISODE_LENGTH_S int 0 basic \
  "Zeitbudget je Episode (Sekunden)" \
  "0 bedeutet 300 s bis zum Auto-Reset, in der Praxis bis zu 9000 Steps — das sind Stunden. Die menschliche Demo dauert 39 s; das Dreifache davon, also 120, ist der uebliche Messwert." \
  --range 0:3600

param DR_ENABLED bool 0 basic \
  "Domain-Randomisierung" \
  "Fuer einen sauberen Messlauf aus (0). An nur, wenn die Robustheit gegen Beleuchtung/Texturen geprueft werden soll." \
  --override "Messlauf ohne Randomisierung; das Skript laesst sie sonst offen"
group "Geschwindigkeit"
note "Gemessen 2026-08-13: 94 % der Zeit gehen in Sim+Rendering, 5 % in die Inferenz. Die Stellschrauben, die wirklich etwas bringen, sitzen deshalb bei den Kameras — nicht beim Modell."

param SCENE_CAM bool 1 advanced \
  "Uebersichtskamera cam_scene rendern" \
  "0 spart eine von fuenf Kameras je Step und aendert die MODELL-EINGABE NICHT — die Policy sieht cam_scene nie. Kostet nur MP4 und Uebersichtsbild. Das ist die einzige gratis Beschleunigung hier."

param CAM_RES_SCALE choice 1 advanced \
  "Kameraaufloesung" \
  "Halbieren geht quadratisch in die Renderzeit ein, aendert aber die Modell-Eingabe. Nur zum Zuschauen, nicht fuer Messlaeufe." \
  --options "1:voll (Messlauf);0.5:halb (nur Zuschauen)"

param EXECUTION_HORIZON int 8 advanced \
  "Aktionen je Inferenz (Chunk-Laenge)" \
  "Wirkt nur auf die 5 % Inferenz — hier fast wirkungslos." \
  --range 1:32

group "Inferenz-Backend"
param GROOT_INFERENCE_BACKEND choice eager advanced \
  "Inferenz-Backend" \
  "TensorRT braucht vorher einen 'optimize all'-Lauf. Details: docs/simulation/inferenz-optimierung.md" \
  --options "eager:Standard;compile:torch.compile;tensorrt:vorher 'optimize all' fahren"

when '[[ "${GROOT_INFERENCE_BACKEND:-eager}" == tensorrt ]]'
param GROOT_TRT_ENGINE_PATH path "" advanced \
  "TensorRT-Engine" \
  "Leer = per Checkpoint-Fingerprint automatisch unter /data/optimized/ finden." \
  --default-from "leer = per Checkpoint-Fingerprint gesucht"
param CAMERA_RENDER_EVERY_N int 1 expert \
  "Nur jedes n-te Frame rendern" \
  "Muss exakt dem EXECUTION_HORIZON entsprechen, sonst sieht die Policy veraltete Bilder. Typisch 8 zusammen mit EXECUTION_HORIZON=8."
