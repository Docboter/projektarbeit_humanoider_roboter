# TL;DR: Menue-Parameter der Aktion tipcheck (Fingerkuppen ins Realbild) von server_rl_run.sh.
action tipcheck "Fingerkuppen aus der FK ins Realbild projizieren" \
  --rank 27 \
  --group "Co-Training vorbereiten" \
  --needs "layout" \
  --hint  "Beantwortet, warum im Render-Lauf keine Kuppe nah an den Wuerfel kommt, obwohl die reale Hand ihn haelt: liegt es an der Kamerapose fuer Realbilder oder an der FK (im 28-dim-State fehlen die Hueftgelenke). Ueber diesen Weg wurden 2026-08-24 drei Geometriefehler gefunden — Vorzeichen der Fingergrundgelenke, Beckenhoehe und Basis in x." \
  --state '[[ -f "${HOST_DATA_DIR:-$HOME/groot-rl-data}/cotrain/layout.json" ]] || echo "! braucht vorher layout"'

param HF_TOKEN secret "" expert "HuggingFace-Token" "Nur noetig, wenn der Datensatz noch fehlt."
param CHECKPOINT_PATH path "/data/checkpoints/groot-g1dex3-checkpoint" expert \
  "Checkpoint im Container" "Nicht noetig — gemessen wird Geometrie, kein Modell."
param HF_CHECKPOINT_REPO str "luca-mue/groot-g1dex3-checkpoint" expert \
  "HuggingFace-Repo des Checkpoints" "Nicht noetig."

group "Umfang"
param TIPCHECK_EPISODES int 4 basic "Anzahl Episoden" --range 1:60
param TIPCHECK_EPISODE_IDS str "" advanced \
  "Nur diese Episoden" "Durch Leerzeichen getrennt, z. B. '0 8 12'." \
  --default-from "project_fingertips_check.py --episode-ids (None)"
param TIPCHECK_FRAME int -1 advanced \
  "Welcher Frame" \
  "-1 = der Bewegungsbeginn aus layout.json. Davor gilt die Ruhelage des Wuerfels, danach traegt die Hand ihn schon." \
  --default-from "project_fingertips_check.py, -1 = Bewegungsbeginn"
group "Wuerfellage"
param RENDER_LAYOUT path "/data/cotrain/layout.json" basic \
  "layout.json aus der Aktion 'layout'" "PFLICHT — liefert Wuerfellage und Pruefframe."
group "Ausgabe"
param TIPCHECK_OUT path "/data/cotrain/tipcheck" advanced "Zielverzeichnis im Container"
param TIPCHECK_APPROACH_STRIDE int 2 expert \
  "Schrittweite des Anflugprofils" "0 schaltet das Profil ab." \
  --default-from "project_fingertips_check.py --approach-stride"
