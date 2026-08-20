# TL;DR: Menue-Parameter der Aktion livecheck (Phase 0 LIVE-Variante) von server_rl_run.sh.
action livecheck "Phase 0 der LIVE-Variante pruefen" \
  --rank 20 \
  --group "Ansehen" \
  --hint  "NVENC, Livestream-Extension, Isaac-Sim-Version und Port-Veroeffentlichung — vor dem ersten LIVESTREAM=2-Lauf. Die LIVE-Variante ist gebaut, aber auf dieser Hardware noch nicht verifiziert; das hier ist der erste Schritt."

param HF_TOKEN secret "" expert "HuggingFace-Token" "Nicht noetig."
param LIVESTREAM_PORT int 49100 advanced "Signaling-Port (TCP)"
param LIVESTREAM_MEDIA_PORT int 47998 advanced "Medien-Port (UDP)" \
  "Der haeufigste Fehler: TCP ist offen, UDP nicht — der Client verbindet, das Bild bleibt schwarz."
