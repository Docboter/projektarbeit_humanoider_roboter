# TL;DR: Menue-Parameter von Training/update_image.sh (welche GR00T-Generation ins Image kommt).
# image-build.spec — gefragt nur, wenn update_image.sh ohne --groot und ohne GROOT_VERSIONS
# am Terminal startet. Eigener Prefix "image", damit die Aktion NICHT in der Aktionsliste
# von setup_and_train_dockerhub_pull.sh (Prefix train) auftaucht.
action build "Trainings-Image bauen und pushen" \
  --cli "" \
  --rank 10 --group "Image" \
  --hint "Baut das Trainings-Image fuer EINE GR00T-Generation und pusht es als :<branch>-<gen> und :<zeitstempel>-<gen>. :latest (N1.6) bzw. :latest-n17 nur mit --push-latest."

# _common.spec fragt den HF-Token — fuer einen Build braucht es ihn nicht.
param HF_TOKEN secret "" expert "HuggingFace-Token (fuer den Build nicht noetig)" \
  --default-from "Geheimnis - hat per Definition keinen Default"

group "GR00T-Generation"
param GROOT_VERSIONS choice 1.6 basic \
  "GR00T-Generation im Image" \
  "Ein Image enthaelt standardmaessig genau EINE Generation. 1.6 ist der bisherige, getestete Weg und behaelt das Tag :latest. 1.7 wird als :latest-n17 getaggt (Python 3.12, torch 2.9, Cosmos-Reason2-2B-Backbone). both packt beide venvs in ein Image — etwa doppelte Groesse und Bauzeit, nur noetig, wenn ein Container zwischen den Generationen wechseln soll." \
  --options "1.6:N1.6 (Default, :latest);1.7:N1.7 (:latest-n17);both:beide venvs (:latest-n16-n17, doppelte Groesse)"
