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
param DOCKER_NAMESPACE str "lucam03" advanced "Docker-Hub-Konto der Images" \
  "Von welchem Konto gezogen wird. lucam03 ist das Team-Konto (oeffentlich, jeder darf ziehen). Wer mit Training/update_image.sh selbst baut und auf sein Konto pusht, traegt das hier ein — am besten dauerhaft in .env.local, dann passen Build und Start automatisch zusammen."
param DOCKER_HUB_IMAGE str '$DOCKER_NAMESPACE/projekt-humanoider-roboter:latest' expert "Docker-Image" \
  "Ganzer Image-Name, schlaegt DOCKER_NAMESPACE. Leer = nach GROOT_VERSION: 1.6 -> :latest, 1.7 -> :latest-n17 (ein Image je Generation, gebaut mit Training/update_image.sh --groot=…). Nach dem Pull prueft der Launcher per Label de.humrob.groot-versions, ob die Generation im Image steckt."
