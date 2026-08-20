# TL;DR: Menue-Parameter der Aktion gap (Domain-Gap real vs. sim) von server_rl_run.sh.
action gap "Domain-Gap real vs. sim messen" \
  --rank 20 \
  --group "Messen" \
  --needs "cams" \
  --hint  "Kosinus-Abstand je Policy-Kamera ueber ein eingefrorenes SigLIP-ViT. Sweep-Varianten (RL_DOME_SWEEP) werden automatisch mitgemessen." \
  --state 'ls "${HOST_DATA_DIR:-$HOME/groot-rl-data}"/cams/*.png >/dev/null 2>&1 || echo "! braucht vorher cams"'

param RL_DOME_SWEEP str "<aus>" advanced \
  "Sweep ueber Dome-Light-Intensitaeten" \
  "Kommagetrennte Werte. Leer = kein Sweep, nur die aktuelle Beleuchtung."
