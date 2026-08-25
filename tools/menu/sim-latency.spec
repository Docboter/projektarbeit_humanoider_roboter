# TL;DR: Menue-Parameter der Aktion latency (reine Policy-Latenz) von server_rl_run.sh.
action latency "Reine Policy-Latenz (ms je Action-Chunk)" \
  --rank 60 \
  --group "Messen" \
  --hint  "In-process, ohne Sim und ohne ZMQ. Die einzige hier messbare Zahl, die auch auf echter Hardware gilt — dort faellt das Rendering weg, das in der Eval 94 % der Zeit frisst."

param LATENCY_ITERS int 50 basic "Anzahl Messungen" --range 1:1000
param EXECUTION_HORIZON int 8 advanced "Chunk-Laenge (nur als Budget-Bezug)"
param GROOT_INFERENCE_BACKEND choice eager basic "Inferenz-Backend" \
  --options "eager:Standard;compile:torch.compile;tensorrt:vorher 'optimize all' fahren"
