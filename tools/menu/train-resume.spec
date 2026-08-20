action resume "Bestehenden Container weiterlaufen lassen" \
  --cli "--resume" \
  --rank 20 --group "Training" \
  --hint  "Daten und Checkpoints bleiben erhalten. Das ist der normale Weg nach einem 'docker stop' oder einem Ctrl-C — nur 'docker rm' zerstoert wirklich etwas." \
  --state 'docker ps -a --format "{{.Names}}" 2>/dev/null | grep -qx "${CONTAINER_NAME:-groot-train}" && echo "✓ Container vorhanden" || echo "! kein Container da"'

param HF_TOKEN      secret "" expert "HuggingFace-Token" "Beim Fortsetzen nicht noetig — er steckt schon im Container."
param WANDB_API_KEY secret "" expert "Weights-&-Biases-Key"
