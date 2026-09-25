# TL;DR: Menue-Parameter der Aktion setup (Checkpoint + USD-Asset laden) von server_rl_run.sh.
action setup "BC-Checkpoint + USD-Asset von HF laden" \
  --rank 20 \
  --group "Vorbereiten" \
  --hint  "Einmalig, ~10 GB. Danach liegen Checkpoint und Roboter-USD unter dem Host-Datenverzeichnis und ueberleben jedes 'clean'." \
  --state '[[ -d "${HOST_DATA_DIR:-$HOME/groot-rl-data}/checkpoints" ]] && echo "✓ liegt bereits vor"'

group "Quelle"
# Beide Felder werden hier erneut genannt, obwohl _common-sim.spec sie schon liefert:
# nur so stehen Repo und Zielpfad in dieser Aktion nebeneinander statt in zwei Gruppen.
param HF_CHECKPOINT_REPO str "luca-mue/groot-g1dex3-checkpoint" advanced \
  "HuggingFace-Repo des Checkpoints" \
  "Nur aendern, wenn ein anderer Trainingslauf ausgewertet werden soll — dann zusammen mit CHECKPOINT_PATH, sonst laedt der zweite Lauf in dasselbe Verzeichnis und findet es als 'bereits vorhanden' vor."
param CHECKPOINT_PATH path "/data/checkpoints/groot-g1dex3-checkpoint" advanced \
  "Zielverzeichnis im Container" \
  "Liegt dort bereits ein VOLLSTAENDIGER Checkpoint, laedt setup nichts nach — das ist die Wiederaufnahme-Logik, aber auch die Falle beim Wechsel des Trainingslaufs: fuer einen zweiten Lauf einen anderen Namen eintragen, sonst laedt er in dasselbe Verzeichnis. Ein angeschnittener Download wird erkannt und fortgesetzt. Ueberlebt 'clean', weil es unter HOST_DATA_DIR/checkpoints/ auf dem Host liegt." \
  --suggest '_menu_suggest_checkpoint'
param BLACK_HANDS bool 1 advanced \
  "Schwarzhaendiges USD verwenden" \
  "Der Domain-Gap-Fix vom Juni: im echten Datensatz sind die Haende schwarz. 0 nimmt das Original-Asset und rendert weisse Haende gegen schwarze im Datensatz."
