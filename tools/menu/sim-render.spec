# TL;DR: Menue-Parameter der Aktion render (Co-Training-Datensatz) von server_rl_run.sh.
action render "Gerenderten Co-Training-Datensatz erzeugen" \
  --rank 30 \
  --group "Co-Training vorbereiten" \
  --needs "layout" \
  --hint  "Schritt 4: echte Dataset-Aktionen in der Sim abspielen und dabei die vier Policy-Kameras aufzeichnen. Ergebnis ist ein LeRobot-v2.1-Datensatz, den run_finetuning_cotrain.sh dazumischt. Zwei Stufen (scan -> render), fortsetzbar, ~2-3 min je Episode. Zurueckgehaltene Test-Episoden werden nie gerendert." \
  --state '[[ -f "${HOST_DATA_DIR:-$HOME/groot-rl-data}/cotrain/layout.json" ]] || echo "! braucht vorher layout"'

group "Umfang"
param RENDER_EPISODES int 60 basic \
  "Anzahl Episoden" \
  "Fuer den Rauchtest 2 zusammen mit RENDER_MAX_FRAMES=60; der echte Lauf nimmt 60." \
  --range 1:500
param RENDER_STAGE choice both basic \
  "Welche Stufe" \
  "'scan' sucht je Episode den Greifpunkt (billig, Kameras auf 1/10). 'render' setzt die Wuerfel dorthin und schreibt in der kalibrierten Aufloesung 640x480." \
  --options "both:beide nacheinander;scan:nur suchen;render:nur schreiben"
param RENDER_MAX_FRAMES int 0 basic \
  "Frames je Episode begrenzen" \
  "0 = ganze Episode. 60 ist die uebliche Rauchtest-Groesse."
param RENDER_EPISODE_IDS str "" advanced \
  "Nur diese Episoden" \
  "Durch Leerzeichen getrennt, z. B. '0 4 8'. Leer = fortlaufend ab 0." \
  --default-from "render_cotrain_dataset.py --episode-ids (None)"
group "Wuerfellage"
param RENDER_LAYOUT path "/data/cotrain/layout.json" basic \
  "layout.json aus der Aktion 'layout'" \
  "PFLICHT. 'none' erzwingt den alten Greifpunkt-Weg — nur als Vergleichslauf sinnvoll, denn dort greift der Arm bei knapp der Haelfte der Episoden ins Leere."

param RENDER_CUBE_SOURCE choice layout advanced \
  "Woher die Wuerfellage kommt" \
  "'grasp' zieht zusaetzlich den GEGRIFFENEN Wuerfel auf den Kuppen-Schwerpunkt der Hand, damit der Griff im Bild aufgeht. Die uebrigen bleiben auf ihrer Bild-Lage." \
  --options "layout:nur aus dem Realbild;grasp:gegriffenen Wuerfel zusaetzlich ankern" \
  --default-from "render_cotrain_dataset.py --cube-source"
param RENDER_MAX_ANCHOR_SHIFT float 0.08 expert \
  "Groesste erlaubte Ankerverschiebung (m)" \
  "Darueber wird die Episode verworfen statt geraten. Zwei Quellen, die weit auseinanderliegen, bezeugen einander nicht." \
  --default-from "render_cotrain_dataset.py --max-anchor-shift"
param RENDER_IGNORE_YAW bool 0 expert \
  "Gierwinkel ignorieren, alle Wuerfel achsparallel" \
  "Nur fuer den A/B-Vergleich: derselbe Episodensatz einmal mit und einmal ohne Drehung. Ohne ihn ist nicht zu trennen, ob eine Verbesserung vom Winkel kommt oder von der Episodenauswahl." \
  --default-from "render_cotrain_dataset.py --ignore-yaw"

group "Schnitt"
param RENDER_STOP_AT_GRASP bool 1 advanced \
  "Jede Episode am ersten Zugreifen abschneiden" \
  "Ab dort entscheidet die Kontaktphysik ueber die Wuerfellage, und das Bild zeigt etwas anderes, als die Aktion beschreibt. Genau solche Paare will das Co-Training nicht lernen."
param RENDER_GRASP_WINDOW int 0 advanced \
  "Nur die letzten N Frames vor dem Griff" \
  "0 = ab Frame 0."
param RENDER_MIN_WINDOW int 60 advanced "Zu kurze Fenster ueberspringen" \
  --default-from "render_cotrain_dataset.py --min-window"
group "Ausgabe"
param RENDER_OUT path "/data/cotrain/g1_dex3_rendered" advanced "Zielverzeichnis im Container"
param RENDER_OVERWRITE bool 0 advanced "Vorhandenes Ergebnis ueberschreiben" \
  "Aus lassen — der Lauf ist fortsetzbar, und Ueberschreiben wirft Stunden Rechenzeit weg."
param DR_ENABLED bool 1 advanced \
  "Domain-Randomisierung" \
  "Beim Rendern AN: die Vielfalt ist hier erwuenscht, sie ist der halbe Zweck des Co-Trainings." \
  --override "beim Rendern erwuenscht — Vielfalt ist der halbe Zweck" \
  --default-from "g1_dex3_blockstack_env.py"
