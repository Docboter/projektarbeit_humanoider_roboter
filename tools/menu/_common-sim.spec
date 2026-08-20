# TL;DR: Gemeinsame Live-Ansicht-Parameter aller Sim-Aktionen von server_rl_run.sh.
# _common-sim.spec — Parameter, die sich die Sim-Aktionen teilen.
# Geladen nach _common.spec, nur fuer den Prefix "sim".

group "Live-Ansicht"
param LIVESTREAM choice 0 advanced \
  "Isaac-Sim-Viewport per WebRTC streamen (Spur A)" \
  "Statt hinterher MP4s zu holen, streamt Isaac Sim seinen 3D-Viewport. Geoeffnet wird er vom nativen 'Isaac Sim WebRTC Streaming Client'. Genau ein Zuschauer, braucht UDP. Vorher einmal 'livecheck' fahren. Anleitung: docs/simulation/live-ansicht.md" \
  --options "0:aus (Default);2:an, privates Netz/VPN — auf diesem Server der richtige Wert;1:an, oeffentliches Netz (vast.ai, ungeschuetzt)"

param LIVE_VIEW bool 0 advanced \
  "MJPEG-Bilder im Browser (Spur B)" \
  "Zustandslos, beliebig viele Zuschauer, per 'ssh -L' tunnelbar. Fuer den tagelangen RL-Lauf gedacht. Kostet keinen zusaetzlichen Renderdurchgang."

param LIVE_KEEP_VIDEO bool 0 advanced \
  "Zusaetzlich MP4s schreiben" \
  "Default ist live STATT Video. 1 = beides."

param LIVESTREAM_PORT      int 49100 expert "WebRTC-Signaling-Port (TCP)" \
  "Auf vast.ai auf den extern gemappten Port setzen (intern==extern), sonst SDP-Port-Mismatch."
param LIVESTREAM_MEDIA_PORT int 47998 expert "WebRTC-Medien-Port (UDP)" \
  "Muss durch die Firewall — sonst verbindet der Client, aber das Bild bleibt schwarz."
param LIVE_VIEW_PORT       int 8900  expert "HTTP-Port des MJPEG-Streams"
param LIVE_VIEW_EVERY_N    int 1     expert "Nur jedes n-te Frame veroeffentlichen" \
  --default-from "live_view.py --live-view-every-n"
param LIVE_VIEW_CAMS       str "cam_left_high,cam_left_wrist" expert "Kameras der Live-Ansicht" \
  "Default sind die kalibrierten Policy-Kameras, also die echte Modell-Eingabe. cam_scene ist eine unvalidierte Uebersichtskamera — mit 'cams' diagnostizieren." \
  --default-from "live_view.py / rl_finetune.py"
group "Verhalten bei Fehlern"
param SHELL_ON_ERROR bool 0 advanced \
  "Bei Fehler in eine Shell im Container fallen" \
  "Empfohlen beim ersten Lauf auf einer neuen Maschine: der Container bleibt stehen und man sieht sich den Zustand an, statt ihn zu verlieren."

group "GR00T-Generation"
param GROOT_VERSION choice auto advanced \
  "GR00T-Generation im Container" \
  "auto liest model_type aus der config.json des Checkpoints — der Normalfall, weil der Checkpoint selbst weiss, womit er trainiert wurde. Nur festnageln, wenn die Erkennung danebengreift. ACHTUNG: Der Wert wird beim ANLEGEN des Containers gesetzt; nach einem Wechsel einmal 'clean' fahren. Host-seitige Aktionen (span, latency, gap) arbeiten bei auto mit 1.6, weil auf dem Host kein Checkpoint liegt. N1.7 ist bislang ungetestet, und rl/baseline/optimize gibt es nur mit 1.6." \
  --options "auto:aus dem Checkpoint erkennen (Default);1.6:Python 3.10, Eagle-Backbone;1.7:Python 3.12, Cosmos-Reason2-2B (gated, ungetestet)" \
  --override "auto = Erkennung aus der Checkpoint-config.json; die 1.6 in env-vars.md gilt dem Training, wo kein Checkpoint zum Erkennen bereitliegt"
