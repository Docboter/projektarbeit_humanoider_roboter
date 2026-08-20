action view "Szene ansehen - ohne Modell, ohne Checkpoint" \
  --rank 10 \
  --group "Ansehen" \
  --hint  "Der einzige Lauf, der ganz ohne Gewichte auskommt: kein HF_TOKEN, kein Checkpoint. Baut nur die Isaac-Lab-Env auf, der Roboter haelt seine Home-Pose. Der billigste Weg zum ersten Bild und der richtige Test fuer die LIVE-Variante, BEVOR ein echter Lauf davon abhaengt. Misst per Konstruktion nichts."

param HF_TOKEN secret "" expert "HuggingFace-Token" "Fuer 'view' bewusst nicht noetig."

group "Laufumfang"
param VIEW_DURATION_S int 3600 basic \
  "Wie lange zusehen (Sekunden)" \
  "Immer > 0. Fuer eine laengere Sitzung ruhig 14400 (4 h) setzen — der Lauf blockiert das Terminal so lange."
param VIEW_NUM_ENVS int 1 advanced "Anzahl paralleler Szenen" --range 1:64
param EPISODE_LENGTH_S int 0 advanced \
  "Sekunden bis zum Auto-Reset" \
  "0 = 300 s. Beim Reset bekommen die Wuerfel neue Positionen."

group "Geschwindigkeit"
param SCENE_CAM bool 0 advanced \
  "Uebersichtskamera cam_scene rendern" \
  "Bei 'view' bewusst 0: die Aktion misst nichts, darf also sparen." \
  --override "do_view spart cam_scene, weil kein MP4 geschrieben wird"
param CAM_RES_SCALE float 0.5 advanced \
  "Kameraaufloesung skalieren" \
  "Bei 'view' bewusst 0.5. Pixel gehen quadratisch in die Renderzeit ein." \
  --override "do_view halbiert die Aufloesung; in jedem Messlauf verboten"
param DR_ENABLED bool 0 advanced "Domain-Randomisierung" "Bei 'view' bewusst aus." \
  --override "do_view braucht keine Randomisierung"
group "Live-Ansicht"
param LIVESTREAM choice 2 basic \
  "Isaac-Sim-Viewport streamen" \
  "Ohne LIVESTREAM/LIVE_VIEW setzt 'view' automatisch LIVESTREAM=2 — sonst saehe man nichts." \
  --options "2:an, privates Netz/VPN (Default hier);1:an, oeffentliches Netz;0:aus" \
  --override "do_view setzt LIVESTREAM=2 selbst, wenn nichts gesetzt ist"
