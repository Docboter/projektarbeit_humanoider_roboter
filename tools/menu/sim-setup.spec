action setup "BC-Checkpoint + USD-Asset von HF laden" \
  --rank 20 \
  --group "Vorbereiten" \
  --hint  "Einmalig, ~10 GB. Danach liegen Checkpoint und Roboter-USD unter dem Host-Datenverzeichnis und ueberleben jedes 'clean'." \
  --state '[[ -d "${HOST_DATA_DIR:-$HOME/groot-rl-data}/checkpoints" ]] && echo "✓ liegt bereits vor"'

group "Quelle"
param HF_CHECKPOINT_REPO str "luca-mue/groot-g1dex3-checkpoint" advanced \
  "HuggingFace-Repo des Checkpoints" \
  "Nur aendern, wenn ein anderer Trainingslauf ausgewertet werden soll."
param BLACK_HANDS bool 1 advanced \
  "Schwarzhaendiges USD verwenden" \
  "Der Domain-Gap-Fix vom Juni: im echten Datensatz sind die Haende schwarz. 0 nimmt das Original-Asset und rendert weisse Haende gegen schwarze im Datensatz."
