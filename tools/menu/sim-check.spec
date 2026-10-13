# TL;DR: Menue-Parameter der Aktion check (LIVE-CHECK ohne Training) von server_rl_run.sh.
action check "LIVE-CHECK: Aufbau ohne Training" \
  --rank 30 \
  --group "Vorbereiten" \
  --needs "setup" \
  --hint  "Baut Env, Policy und Critic mit 2 Envs auf und trainiert NICHT. Der reale Nachweis, dass Rendering und Env-Konstruktion auf dieser GPU laufen — vor dem ersten langen Lauf."

group "Live-Ansicht"
param LIVESTREAM choice 0 advanced "Isaac-Sim-Viewport streamen" \
  --options "0:aus;2:an, privates Netz/VPN;1:an, oeffentliches Netz"
