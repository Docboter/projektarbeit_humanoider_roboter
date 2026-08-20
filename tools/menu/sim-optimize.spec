action optimize "DiT nach ONNX + TensorRT-Engine bauen" \
  --rank 10 \
  --group "Beschleunigen" \
  --needs "setup" \
  --hint  "Exportiert, baut die GPU-spezifische Engine und validiert sie. 'all' fuehrt zusaetzlich einen kurzen Closed-Loop-A/B-Lauf aus und schreibt benchmark_report.json unter /data/optimized/<checkpoint-fingerprint>/. Danach ist GROOT_INFERENCE_BACKEND=tensorrt benutzbar."

argpos 2 all "Phase" \
  "'all' ist der uebliche Weg. Die Einzelphasen sind zum Nachfassen da, wenn ein Schritt fehlgeschlagen ist." \
  --options "all:alles nacheinander (Default);export:nur ONNX-Export;build:nur Engine bauen;validate:nur pruefen;benchmark:nur messen"

param OPTIMIZE_ITERATIONS int 20 advanced "Messiterationen" --range 1:1000
param OPTIMIZE_WARMUP int 5 advanced "Aufwaermdurchlaeufe"
param OPTIMIZE_SIM_EPISODES int 1 advanced "Episoden im A/B-Lauf"
param OPTIMIZE_SIM_EPISODE_LENGTH_S int 20 advanced "Sekunden je A/B-Episode"
param OPTIMIZE_WORKSPACE_MB int 8192 expert "TensorRT-Workspace (MB)"
param OPTIMIZE_TARGET_SPEEDUP float 2.0 expert "Erwarteter Faktor" \
  "Wird der nicht erreicht, meldet der Lauf das — die Engine ist dann meist fuer die falsche GPU gebaut."
