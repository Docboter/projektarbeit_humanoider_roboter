action destroy "Container loeschen und neu starten" \
  --cli "--destroy" \
  --rank 30 --group "Training" \
  --hint  "ACHTUNG: Alle Daten und Checkpoints im Container sind danach weg — es gibt keinen Host-Mount. Vorher sichern: docker cp <container>:/data/g1_dex3_finetune ./checkpoints" \
  --state 'docker ps -a --format "{{.Names}}" 2>/dev/null | grep -qx "${CONTAINER_NAME:-groot-train}" && echo "✓ Container vorhanden"'

param HF_TOKEN secret "" basic "HuggingFace-Token" \
  --default-from "Geheimnis - hat per Definition keinen Default"
