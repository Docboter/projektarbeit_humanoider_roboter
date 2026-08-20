action webview "Browser-Client fuer den WebRTC-Viewport" \
  --rank 30 \
  --group "Ansehen" \
  --hint  "Eigener kleiner Container ohne Simulator, der nur eine Seite serviert; der Browser verbindet sich direkt auf die Ports von groot-rl. Spart die App-Installation, NICHT den UDP-Port. Zeigt allein nichts — es braucht PARALLEL einen Lauf mit LIVESTREAM=2. Nur Chromium/Chrome/Edge."

argpos 2 "" "Unterbefehl" \
  "Leer startet den Web-Viewer, 'stop' beendet ihn." \
  --options ":starten;stop:beenden"

param HF_TOKEN secret "" expert "HuggingFace-Token" "Nicht noetig."
param WEBVIEW_PORT int 8210 advanced "Host-Port der Seite"
