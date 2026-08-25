# TL;DR: Gemeinsame Trainings-Parameter (WANDB_API_KEY, Container-Name/-Image).
# _common-train.spec — Parameter, die sich die Trainings-Aktionen teilen.
# Geladen nach _common.spec, nur fuer den Prefix "train".

group "Zugang"
param WANDB_API_KEY secret "" basic \
  "Weights-&-Biases-Key" \
  "Leer lassen laeuft ohne W&B. Bei einem Lauf ueber Stunden oder Tage ist das Dashboard aber der einzige bequeme Weg, den Verlauf zu sehen — und ohne ihn faellt die Kurve weg, an der man einen abgebrochenen Lauf spaeter beurteilt." \
  --default-from "Geheimnis - hat per Definition keinen Default"
group "Container"
param CONTAINER_NAME str "groot-train" expert "Name des Trainings-Containers"
param DOCKER_HUB_IMAGE str "lucam03/projekt-humanoider-roboter:latest" expert "Docker-Image"
