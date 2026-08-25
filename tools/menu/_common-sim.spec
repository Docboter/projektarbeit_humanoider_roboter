# TL;DR: Gemeinsame Gewichts- und Live-Ansicht-Parameter aller Sim-Aktionen von server_rl_run.sh.
# _common-sim.spec — Parameter, die sich die Sim-Aktionen teilen.
# Geladen nach _common.spec, nur fuer den Prefix "sim".

# Welche Gewichte gemessen werden, ist bei einem Messlauf die wichtigste Angabe
# ueberhaupt. Bis 2026-08-21 stand sie in genau EINER Spec (sim-setup.spec) — jede
# andere Aktion ruft ensure_checkpoint aber selbst auf und hat den Default-Checkpoint
# im Zweifel stillschweigend heruntergeladen, ohne je danach zu fragen. Deshalb hier,
# geteilt von allen Aktionen, die Gewichte laden.
#
# Aktionen, die ohne Gewichte auskommen (preflight, view, webview, livecheck, gap,
# layout, layoutcheck, shell, clean) stufen beide Felder auf "expert" zurueck — dasselbe
# Muster, mit dem HF_TOKEN aus _common.spec bei "view" verschwindet.
group "Gewichte"
param CHECKPOINT_PATH path "/data/checkpoints/groot-g1dex3-checkpoint" advanced \
  "Checkpoint im Container" \
  "Der eigentliche Hebel. ensure_checkpoint prueft, ob unter diesem Pfad ein VOLLSTAENDIGER Checkpoint liegt (config.json + *.safetensors, keine .incomplete-Reste): liegt er da, wird nichts nachgeladen und HF_CHECKPOINT_REPO ist wirkungslos. Ein zweiter Trainingslauf braucht deshalb auch einen anderen Pfad, sonst misst man wieder den alten. Der Pfad gilt im Container; auf dem Host liegt er unter HOST_DATA_DIR/checkpoints/. Das USD-Asset muss NICHT mitwandern — es ist fuer alle Laeufe dasselbe und wird notfalls an den anderen bekannten Orten gesucht." \
  --suggest '_menu_suggest_checkpoint'

param HF_CHECKPOINT_REPO str "luca-mue/groot-g1dex3-checkpoint" advanced \
  "HuggingFace-Repo des Checkpoints" \
  "Woher geladen wird, falls CHECKPOINT_PATH im Container noch fehlt (~10 GB, einmalig). Nur aendern, wenn ein anderer Trainingslauf ausgewertet werden soll — dann zusammen mit CHECKPOINT_PATH."

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
