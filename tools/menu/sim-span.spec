# TL;DR: Menue-Parameter der Aktion span (Fingerspanne, echte Bilder) von server_rl_run.sh.
action span "Fingerspanne auf ECHTEN Datensatz-Bildern" \
  --rank 50 \
  --group "Messen" \
  --hint  "Offene Schleife, kein Isaac Sim. Trennt 'Domain-Gap' von 'Modell hat den Griff nie gelernt' — das Gate vor einem TUNE_VISUAL-Lauf. Holt den echten Datensatz bei Bedarf selbst (~18 GB, einmalig)."

group "Datensatz"
param SPAN_TRAJ_IDS str "0 1 2 3 4" basic \
  "Trajektorien-IDs" \
  "Durch Leerzeichen getrennt. Mehr IDs = stabilere Zahl, laengere Laufzeit."
param SPAN_AUTO_FETCH bool 1 advanced \
  "Datensatz bei Bedarf selbst holen" \
  "0, wenn der Datensatz schon liegt und der Download-Check nur Zeit kostet."
param SPAN_DATASET path "/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset" expert \
  "Pfad des Datensatzes im Container"
