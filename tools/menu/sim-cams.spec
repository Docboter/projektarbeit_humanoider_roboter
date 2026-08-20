action cams "Kamera-Diagnose: Pose + ein PNG je Kamera" \
  --rank 10 \
  --group "Messen" \
  --hint  "Vergleicht die KONFIGURIERTE mit der tatsaechlich gerenderten Pose und legt ein Bild je Kamera ab. In Lauf 13 lagen beide 95,6 Grad auseinander und drei Laeufe waren umsonst — diese Aktion ist die Versicherung dagegen."

param RL_NUM_ENVS int 4 advanced "Anzahl Envs beim Dumpen" --range 1:64 \
  --override "Diagnoselauf braucht nur wenige Envs"
param RL_SETTLE_STEPS int 8 expert "Steps, bevor die Pose gelesen wird" \
  "Die Szene muss sich erst setzen; direkt nach dem Reset stehen die Kameras noch nicht."
