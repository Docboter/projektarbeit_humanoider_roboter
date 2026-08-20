action layoutcheck "Kameramodell gegen ein Renderbild pruefen" \
  --rank 10 \
  --group "Co-Training vorbereiten" \
  --hint  "Bevor man 'layout' glaubt. In Lauf 13 lagen konfigurierte Pose und cam.data 95,6 Grad auseinander und drei Laeufe waren umsonst."

param HF_TOKEN secret "" expert "HuggingFace-Token" "Nicht noetig."

group "Eingaben"
param LAYOUTCHECK_FRAME path "" basic \
  "Gerendertes Bild im Container" \
  "Leer = das Skript sucht selbst. Sonst der Pfad eines Bildes aus einem 'render'- oder 'cams'-Lauf." \
  --default-from "extract_block_layout.py, leer = selbst suchen"
param LAYOUTCHECK_EXPECT str "" basic \
  "Bekannte Wuerfelpositionen als JSON" \
  "Aus render_manifest.json, Feld cubes_xyz. Ohne diese Referenz kann der Check nichts beweisen." \
  --default-from "ohne Referenz beweist der Check nichts"
param LAYOUTCHECK_CAM str "cam_left_high" advanced "Zu pruefende Kamera"
