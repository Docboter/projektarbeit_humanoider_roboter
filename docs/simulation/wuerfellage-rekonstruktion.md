# Würfellage aus den Realbildern — Verfahren, Koordinaten, offene Punkte

**Stand:** 2026-08-17 · Verfahren gebaut und lokal geprüft, **Abnahme-Test gegen den Renderer
offen** · Vorgeschichte: [co-training.md](../training/co-training.md) §3.0/§3.2a

Dieses Dokument beschreibt, **wie die Position der drei Würfel aus den realen
Trainingsaufnahmen bestimmt wird** und **wie die daraus gewonnenen Bildkoordinaten in
Sim-Koordinaten überführt werden**. Es ist als Übergabe gedacht: die beiden vorherigen
Anläufe haben das Ziel nicht erreicht, und wer den dritten weiterbaut, sollte wissen, woran
die ersten beiden gescheitert sind und welche Annahme im aktuellen Verfahren als Nächstes
angreifbar ist.

---

## 1. Wozu die Würfellage überhaupt gebraucht wird

Für das Co-Training (Schritt 4) werden Paare *(Sim-Bild, echte Aktion)* erzeugt: die
aufgezeichneten Aktionen einer realen Episode werden in Isaac Lab abgespielt und dabei die
vier Policy-Kameras aufgezeichnet. Der Vision-Encoder soll auf Real- und Sim-Bildern
dieselbe Aktion produzieren müssen und sich dadurch nicht mehr auf die Oberflächen-Statistik
der Realbilder verlassen können.

Das funktioniert nur, wenn das Sim-Bild zeigt, was die Aktion tut. **Der reale Datensatz
enthält aber keine Objektposen** — nur Roboter-State, Aktionen und Videos. Legt die Env ihre
Würfel zufällig aus, entstehen Paare, in denen der Arm dorthin greift, wo kein Würfel liegt;
der Encoder lernt daraus, den Würfel zu *ignorieren*. Das wäre schlimmer als gar nichts.

Die Würfellage muss also aus dem Datensatz **rekonstruiert** werden. Genau darum geht es hier.

---

## 2. Die beiden vorherigen Versuche — und woran sie scheiterten

### 2.1 Versuch 1: Greifpunkt aus der Fingerkinematik (`scan.json`)

**Idee.** Die Episode einmal billig abspielen (Kameras auf 1/10 Auflösung), je Hand den
Moment der engsten Fingeröffnung suchen und dort den Schwerpunkt der drei Fingerkuppen
nehmen. Dort muss in der echten Aufnahme ein Würfel gelegen haben. Implementiert in
`find_grasp_points` / `place_cubes` in
[`render_cotrain_dataset.py`](../../Simulation/g1_dex3_sim/render_cotrain_dataset.py).

**Warum es nicht trägt.** `np.argmin` sucht das Minimum über die **ganze** Episode. Bei einem
Pick-and-Place bleibt die Hand vom Zugreifen bis zum Ablegen geschlossen — die Öffnungsspur
hat ein **breites Tal, keinen Ausschlag**. Wo innerhalb dieses Tals das Minimum liegt,
entscheidet minimales Nachdrücken, ist also praktisch eine Zufallsstichprobe irgendwo auf dem
Transportweg. Gemessen am Lauf vom 2026-08-17 (60 Episoden, 116 gültige Griffe):

| `close_step` / Episodenlänge | p10 | p25 | Median | p75 | p90 |
|---|---|---|---|---|---|
| | 13 % | 21 % | **50 %** | 78 % | 85 % |

**48 von 116 Griffen liegen jenseits von 60 % der Episode, 21 jenseits von 80 %.** Ein Pick
kann dort nicht sein.

Zwei weitere Fehler kommen dazu:

- **z wird weggeworfen.** `place_cubes` nimmt nur x/y und setzt z auf die Tischauflage.
  Schließt die Hand 15 cm über dem Tisch, landet der Würfel 15 cm *unter* den Fingern.
- **Ein Greifpunkt je Hand.** „Stack three block" braucht zwei bis vier Pick-and-Place-Zyklen.
  Für jeden Griff außer einem pro Hand liegt kein Würfel — und Würfel 2 wurde ohnehin
  zufällig „daneben" gelegt.

Beobachtbare Folge: der Arm greift ins Leere, in x, y **und** z.

### 2.2 Versuch 2: Nur bis zum Griff rendern (`--stop-at-grasp`)

**Idee.** Frames ab dem Griff sind falsch beschriftet, weil ab dort die Kontaktphysik über
die Würfellage entscheidet — und die greift im Replay meist gar nicht: bei **101 von 116
Griffen** bleibt die engste erreichte Kuppenöffnung über 6 cm, bei 5 cm Würfelkante. Also die
Episode am frühesten `close_step` abschneiden.

**Warum es nicht reicht.** Es repariert die Beschriftung *nach* dem Griff, sagt aber nichts
darüber, ob der Würfel an der richtigen Stelle liegt. Liegt er falsch, sind auch die
Anfahrt-Frames falsch. Der Schnitt ist weiterhin sinnvoll (er bleibt im Renderer), aber er
war nie der Kern des Problems.

### 2.3 Was beide gemeinsam haben

Beide versuchen, die Würfellage aus der **Roboterbewegung** zu erschließen. Die Bewegung
enthält diese Information nur indirekt und mehrdeutig. Der Datensatz enthält sie direkt —
**im Bild**.

---

## 3. Das aktuelle Verfahren

Implementiert in
[`extract_block_layout.py`](../../Simulation/g1_dex3_sim/extract_block_layout.py) (Detektion,
Rückprojektion) und [`camera_geometry.py`](../../Simulation/g1_dex3_sim/camera_geometry.py)
(Kameramodell). Beide brauchen **kein Isaac Lab und keine GPU** — das ist Absicht, siehe §7.

```
Realvideo Episode n, Frame 0
   │
   ├─ (a) HSV-Segmentierung rot / grün / gelb           → binäre Masken
   ├─ (b) größte zusammenhängende Fläche je Farbe       → Blob
   ├─ (c) Schwerpunkt des Blobs                         → Pixel (u, v)
   ├─ (d) Pixel → Kamerastrahl → Schnitt mit z = 0.915  → (x, y) env-lokal
   ├─ (e) Bias-Korrektur (aus dem Abnahme-Test)         → (x, y) korrigiert
   └─ (f) Mittel über beide High-Kameras                → layout.json
                                                             │
   render_cotrain_dataset.py --layout  →  place_cubes  ←──────┘
```

### (a) Farbsegmentierung

Die drei Würfel sind rot, grün und gelb; der Tisch ist weiß, die Hände sind schwarz, die Arme
grau. In HSV trennt das sauber: Tisch und Arme scheitern an der Sättigung, die Hände am
Helligkeitswert.

| Würfel | Env-Objekt | Farbton H | S ≥ | V ≥ |
|---|---|---|---|---|
| rot | `block_0` | < 15° oder > 345° | 0,40 | 0,18 |
| grün | `block_1` | 75°–175° | 0,28 | 0,15 |
| gelb | `block_2` | 35°–70° | 0,35 | 0,28 |

Die Zuordnung Farbe → `block_i` ist die aus
[`g1_dex3_blockstack_env.py`](../../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py); die
Sim-Farben wurden dort bereits an die Realdaten angeglichen (früher blau statt gelb).

Die RGB→HSV-Umrechnung ist in numpy ausgeschrieben statt aus OpenCV geholt — das Werkzeug
soll auch in einem Container ohne `cv2` laufen.

### (b)/(c) Blob und Schwerpunkt

Pro Maske die größte zusammenhängende Fläche (4-Nachbarschaft, `scipy.ndimage.label` wenn
vorhanden, sonst ein eigener Flood-Fill), Mindestfläche 120 px. Deren Pixelschwerpunkt ist
`(u, v)`.

**Achtung, hier sitzt ein bekannter Bias:** Der Blob ist die Silhouette eines schräg von oben
gesehenen Würfels, also Deckfläche **plus** zugewandte Seitenfläche. Sein Schwerpunkt ist
damit *nicht* die Projektion des Würfelmittelpunkts, sondern liegt zur Kamera hin verschoben.
Siehe (e) und §8.2.

### (d) Rückprojektion — siehe §4

### (e) Bias-Korrektur

Eine additive Verschiebung `(dx, dy)` in Metern, die auf das Ergebnis der Rückprojektion
addiert wird (`--bias`, bzw. `LAYOUT_BIAS="dx dy"`). Sie wird **gemessen**, nicht geschätzt:
`detect --expect` auf einem gerenderten Sim-Bild mit bekannten Würfelpositionen liefert das
mittlere Residuum, und genau das ist der Bias. Der Rest — die Streuung — bleibt.

### (f) Zwei Kameras

`cam_left_high` und `cam_right_high` liefern je eine Schätzung; gespeichert wird das Mittel,
die Differenz landet als `camera_spread_cm` im `layout.json`.

> **Wichtig, damit die Zahl nicht falsch gelesen wird:** Die Stereobasis beträgt 4,7 cm bei
> ~0,50 m Motivabstand. Ein Fehler im **Kameramodell** verschiebt beide Strahlen fast gleich
> und ist in dieser Differenz praktisch unsichtbar. `camera_spread_cm` prüft die
> **Detektion**, nicht die Pose. Für die Pose gibt es nur den Abnahme-Test aus §7.

---

## 4. Koordinatensysteme und Transformationskette

Das ist der Teil, den man beim Weiterbauen wirklich verstanden haben muss.

### 4.1 Die vier beteiligten Systeme

| System | Achsen / Einheiten | Bemerkung |
|---|---|---|
| **Pixel** | `u` 0…639 nach rechts, `v` 0…479 nach unten | Mitte von Pixel `u` liegt bei `u + 0,5` |
| **Kamera** | X = Blickachse, Y = **links**, Z = **oben** | Isaac-Lab-`convention="world"` |
| **Env-lokal** | x nach vorn, y nach links, z nach oben; Ursprung am Env-Prim | Tisch: x 0,10–0,90, y ±0,30, Oberkante z = 0,87 |
| **Isaac-Welt** | Env-lokal **+** `scene.env_origins[i]` | bei einer einzelnen Env = (0,0,0), wird trotzdem addiert |

### 4.2 Intrinsics

```
f_px = focal_length_mm / horizontal_aperture_mm · W
     = 13,6545 / 20,955 · 640
     = 417,03 px
cx, cy = W/2, H/2 = 320, 240
```

Isaac Lab leitet die vertikale Apertur aus dem Seitenverhältnis ab, die Pixel sind also
quadratisch und `f_px` gilt für beide Achsen. Die Brennweite wiederum kommt aus dem
gewünschten Sichtfeld: `focal = aperture / (2·tan(HFOV/2))` mit HFOV = 75°.

### 4.3 Pixel → Kamerastrahl

```
d_rechts = ((u + 0,5) − cx) / f_px
d_unten  = ((v + 0,5) − cy) / f_px

d_cam = ( 1 , −d_rechts , −d_unten )          # Komponenten entlang (Blick, links, oben)
```

**Warum die beiden Minuszeichen.** Im Kameraframe zeigt Y nach *links* und Z nach *oben*.
Bild-rechts ist damit −Y, Bild-unten ist −Z. Wer diese Vorzeichen vertauscht, bekommt ein in
beiden Achsen gespiegeltes Layout, das an einer symmetrischen Szene lange plausibel aussieht.
(Herleitung aus Isaacs Konventionen: `opengl` hat −Z vorwärts und +Y oben, `world` hat +X
vorwärts und +Z oben; daraus folgt Bild-rechts = +X_gl = −Y_world.)

### 4.4 Kamerastrahl → Env-Koordinaten

```
R = quat_to_matrix(rot)        # Spalten = Kamera-Achsen im Env-Frame
d = R · d_cam
```

`rot` ist das in `G1Dex3CameraCfg` hinterlegte Quaternion `(w, x, y, z)`, erzeugt von
`look_at_world_quat(eye, target)`. Für `cam_left_high` konkret:

```
eye = (0,0537 ; 0,0250 ; 1,3239)          # d435_link, auf die Mittellinie zentriert
rot = (0,88701 ; 0 ; 0,46176 ; 0)         # = 55° Neigung nach unten

        ⎡  0,5736   0   0,8192 ⎤     Spalte 0 = Blickachse  (nach vorn und unten)
R   =   ⎢  0        1   0      ⎥     Spalte 1 = links       (= env +y)
        ⎣ −0,8192   0   0,5736 ⎦     Spalte 2 = oben        (gekippt)
```

`cam_right_high` ist dieselbe Orientierung, `eye` nur um −0,05 m in y versetzt — die beiden
Kameras blicken **parallel**, nicht konvergent.

### 4.5 Strahl → Würfelebene

```
t = (z_ebene − eye_z) / d_z
P = eye + t · d                               mit z_ebene = 0,915
```

`0,915` ist `block_z_surface` aus der Env, also die **Mittelpunktshöhe** eines auf dem Tisch
liegenden Würfels.

> **Kuriosum, das man kennen sollte:** Die Tischoberkante liegt bei z = 0,87, ein 5-cm-Würfel
> mit Mittelpunkt bei 0,915 hat seine Unterkante also bei 0,89 — er schwebt 2 cm über der
> Platte. Das ist in der Env so gewollt (der Tisch wurde abgesenkt, weil das Modell nach
> z ≈ 0,915 greift und die Hände sich bei 0,89 verklemmten). Wer die Ebene ändert, muss beide
> Werte gemeinsam anfassen.

### 4.6 Bias und Mittelung

```
P_korr = P + (dx, dy, 0)
(x, y) = Mittel über die verfügbaren Kameras
```

### 4.7 Env-Koordinate → gesetzter Würfel

In `place_cubes`:

```
world = ( x + origin_x , y + origin_y , block_z_surface , 1, 0, 0, 0 )
block.write_root_pose_to_sim(world)
block.write_root_velocity_to_sim(0)
```

Also: nur x/y stammen aus dem Bild, z ist immer die Tischauflage, die Orientierung ist die
Identität (achsparallel). Danach laufen `--settle-steps` Schritte, bevor aufgezeichnet wird.

### 4.8 Die Rückrichtung — für die Validierung

Projektion eines bekannten Env-Punkts ins Bild:

```
a = Rᵀ · (P − eye)                            # (a_vor, a_links, a_oben)
u = cx + f_px · (−a_links / a_vor) − 0,5
v = cy + f_px · (−a_oben  / a_vor) − 0,5
```

`PinholeCamera.project` / `.backproject_to_plane` sind exakt invers zueinander (numerisch auf
< 1 µm geprüft). **Das beweist allerdings nur die Konsistenz des Modells mit sich selbst,
nicht seine Übereinstimmung mit dem Renderer** — dazu §7.

### 4.9 Auflösungsgrenze

Ein Pixel deckt auf der Würfelebene ab:

| Bildbereich | entspricht x | cm pro Pixel (quer) |
|---|---|---|
| oben (fern) | x ≈ 0,66 | 0,163 |
| Mitte | x ≈ 0,34 | 0,120 |
| unten (nah) | x ≈ 0,14 | 0,092 |

Ein Detektionsfehler von 5 px ist also gut 0,5 cm. Die Genauigkeitsgrenze des Verfahrens
liegt damit nicht bei der Pixelauflösung, sondern bei den Annahmen aus §5.

---

## 5. Die Annahmen — die Liste, die man beim Weiterbauen angreift

| # | Annahme | Status |
|---|---|---|
| A1 | Die drei Würfel liegen in Frame 0 auf dem Tisch | solide (Episodenanfang) |
| A2 | Die Sim-Kamerapose entspricht der realen Kamera | **rekonstruiert, nicht kalibriert** (14 Overlay-Iterationen nach Augenmaß) |
| A3 | Isaac rendert mit der *konfigurierten* Pose | **ungeprüft für den aktuellen Stand** — genau hier lag der Lauf-13-Bug |
| A4 | Blob-Schwerpunkt ≈ Projektion des Würfelmittelpunkts | **systematisch falsch**, Größenordnung 1–2 cm; wird als Bias herausgerechnet |
| A5 | Würfelmittelpunkte liegen auf z = 0,915 | solide, solange sie auf dem Tisch liegen |
| A6 | Die Farben sind trennbar | geprüft, 3/3 in beiden Kameras |
| A7 | Kein Würfel ist in Frame 0 verdeckt | **nicht behandelt** — bei Verdeckung fällt der Würfel stumm aus |

A2 und A3 sind die gefährlichen: beide erzeugen einen **globalen Versatz**, der alle Würfel
gleich verschiebt und deshalb in keiner der bisherigen Proben auffällt.

---

## 6. Was geprüft ist — und was die Probe jeweils beweist

Alles Folgende wurde am 2026-08-17 **lokal** gegen
[`Simulation/camera_reference/`](../../Simulation/camera_reference/) gemessen, ohne Container.

| Probe | Ergebnis | Was sie beweist |
|---|---|---|
| Roundtrip Projektion ↔ Rückprojektion | exakt (< 1 µm) | Modell ist in sich konsistent — **sonst nichts** |
| Bildaufteilung: Tisch hinten / vorn | 1 % / 97 % der Bildhöhe | passt zur dokumentierten Kalibrierung (5 % / 99 %) |
| **Skalenprobe am bekannten 5-cm-Würfel** | **5,2 cm quer** (Mittel über 6 Messungen) | **Intrinsics × Kameraabstand stimmen** — eine bekannte Länge geht rein und kommt richtig heraus |
| Vorhergesagte Kantenlänge im Bild | 41,8 px bei 0,499 m | deckt sich mit den 40–46 px, die 2026-08-08 unabhängig gemessen wurden |
| Detektion | 3/3 Würfel in beiden Kameras, Kreuze auf den Würfeln | Segmentierung trägt |
| Beide Kameras einig bis | 1,3–2,5 cm | Detektionsrauschen, **nicht** die Pose (§3f) |

Gemessene Lagen im Referenzframe (env-lokal, in Metern):

| Würfel | `cam_left_high` | `cam_right_high` | Differenz |
|---|---|---|---|
| rot | (0,327 ; −0,048) | (0,323 ; −0,068) | 2,0 cm |
| grün | (0,355 ; +0,131) | (0,346 ; +0,108) | 2,5 cm |
| gelb | (0,256 ; +0,164) | (0,256 ; +0,151) | 1,3 cm |

### Nebenbefund: das Stapelband

Das schwarze Band ist in der Env mit 12 cm Breite bei (0,35 ; 0,00) konfiguriert — im Code
ausdrücklich als Näherung markiert. Aus den Realbildern zurückgerechnet ergeben sich
**7,2 cm bei (0,31 ; +0,06)**. Damit steht erstmals eine Messung gegen die Schätzung. Für die
Würfellage ist das irrelevant, für den Domain Gap nicht.

---

## 7. Der offene Abnahme-Test — bitte zuerst

**Solange dieser Test nicht gelaufen ist, ist das Layout plausibel und unbelegt.**

Alle bisherigen Proben können einen globalen Versatz nicht sehen. Ein solcher Versatz ist in
diesem Projekt schon einmal teuer geworden: in Lauf 13 (2026-08-08) lagen die konfigurierte
USD-Pose und `cam.data` **95,6°** auseinander, drei Sim-Läufe waren umsonst, und der
Fehlschluss lautete damals „einfarbiges Bild ⇒ kein Blickwinkelproblem".

Der Test schickt ein **gerendertes** Sim-Bild mit **bekannten** Würfelpositionen durch
denselben Detektor:

```bash
D=/home/lmuecke/project/data/RL/cotrain/g1_dex3_rendered
for EP in 0 12 41; do
  EXPECT=$(python3 -c "import json;print(json.dumps(json.load(open('$D/render_manifest.json'))['episodes']['$EP']['cubes_xyz']))")
  LAYOUTCHECK_FRAME=/data/cotrain/g1_dex3_rendered/videos/chunk-000/observation.images.cam_left_high/episode_$(printf %06d $EP).mp4 \
  LAYOUTCHECK_EXPECT="$EXPECT" ./Simulation/server_rl_run.sh layoutcheck
done
```

Auswertung:

- **Mittel** = Bias aus A2 + A3 + A4 zusammen → per `LAYOUT_BIAS="dx dy"` (Meter)
  herausrechnen.
- **Streuung** = der Rest, der bleibt. Über ~2 cm trägt das Layout nicht; dann zuerst A4
  angehen (§8.2), nicht rendern.
- Die markierten Kontrollbilder unter `LAYOUT_DEBUG_DIR` zeigen sofort, ob die Kreuze
  überhaupt auf den Würfeln sitzen.

Ein Vorbehalt zum Test selbst: die Referenzepisoden stammen aus dem alten Lauf, in dem die
Würfel am Greifpunkt lagen. Steht in Frame 0 eine Hand auf einem Würfel, hat der
Kontakt-Solver ihn während der `settle-steps` verschoben und das Residuum ist verfälscht.
Deshalb mehrere Episoden fahren und die Kontrollbilder ansehen, statt einer Zahl zu glauben.

---

## 8. Ansatzpunkte zum Weiterbauen — nach Nutzen sortiert

### 8.1 Abnahme-Test fahren und den Bias eintragen
Siehe §7. Billigster Schritt mit dem größten Erkenntnisgewinn.

### 8.2 Den Schwerpunkts-Bias (A4) direkt beseitigen statt herauszurechnen
Der Bias hängt vom Blickwinkel ab, ist also über das Bild **nicht konstant** — eine additive
Korrektur ist nur die erste Ordnung. Zwei bessere Wege:

- **Deckfläche statt Silhouette.** Die Oberseite ist heller als die Seitenflächen; nur sie
  segmentieren und den Strahl mit z = 0,94 (Würfeloberkante) schneiden. Dann fällt der
  Seitenflächen-Anteil weg.
- **Modellanpassung.** Den bekannten 5-cm-Würfel als Box in die Pose einpassen, die die
  beobachtete Silhouette am besten erklärt (2 Freiheitsgrade x/y, ggf. Gierwinkel als
  dritter). Aufwendiger, liefert aber nebenbei die **Orientierung**, die derzeit als
  achsparallel angenommen wird.

### 8.3 Die Kamerapose echt kalibrieren (A2)
Die aktuelle Pose ist das Ergebnis von 14 Overlay-Iterationen nach Augenmaß. Sauber wäre PnP
aus bekannten Landmarken. Dafür fehlt bislang die **vermessene reale Tischgeometrie** — der
Tisch in der Env ist ein Platzhalter (0,8 × 0,6 × 0,87 m), das Stapelband ebenfalls geschätzt
(und laut §6 um 5 cm zu breit). Ein Zollstock am realen Aufbau ist hier mehr wert als jede
weitere Iteration am Bild.

### 8.4 Verdeckung behandeln (A7)
Derzeit: Würfel nicht gefunden → fällt stumm aus, `place_cubes` fällt auf Greifpunkt oder
Zufall zurück. Besser: den frühesten Frame suchen, in dem alle drei sichtbar sind, und die
zweite Kamera als Ausweichquelle nutzen (die Struktur `per_camera` in `layout.json` hält das
bereits vor).

### 8.5 Layout je Frame statt nur Frame 0
Der eigentliche nächste Schritt für den Renderer. Damit ließe sich
(a) erkennen, wann ein Würfel angestoßen wurde, und
(b) das **kinematische Attach** speisen: Würfelpose zwischen `close` und `release` jeden Step
auf den Kuppen-Schwerpunkt schreiben, bei `release` fallen lassen. Erst das macht Transport-
und Stapelphasen verwendbar und hebt den nutzbaren Anteil des Materials von ~40 % auf ~100 %.
Ohne Attach bleibt `--stop-at-grasp` nötig.

### 8.6 Wrist-Kameras einbeziehen
`PinholeCamera.from_cfg` lässt nur die drei weltfesten Kameras zu. Die Handgelenkskameras
hängen an einem Link, ihre Weltpose ändert sich mit jeder Roboterpose und kann nur aus der
laufenden Sim kommen. Für eine Nahaufnahme des Würfels wären sie interessant — dann muss die
Pose aber pro Frame aus `cam.data` gezogen und der Modellbau umgestellt werden.

---

## 9. Was dieses Verfahren *nicht* löst

Die Würfellage ist nur die **Anfangsbedingung**. Ab dem Moment, in dem die Hand zugreift,
entscheidet die Kontaktphysik über die Würfellage — und die greift im Replay meist nicht
(§2.2). Die Frames ab dem Griff bleiben deshalb falsch beschriftet, unabhängig davon, wie
genau das Layout ist. Dafür ist §8.5 zuständig.

Ebenfalls offen: die **Orientierung** der Würfel. Aktuell werden sie achsparallel gesetzt; im
Realbild sind sie erkennbar verdreht.

---

## 10. Dateien, Befehle, Kennzahlen

### Dateien

| Datei | Rolle |
|---|---|
| [`camera_geometry.py`](../../Simulation/g1_dex3_sim/camera_geometry.py) | Posen, Intrinsics, `PinholeCamera.project` / `.backproject_to_plane`. **Isaacfrei** — nur deshalb außerhalb des Containers prüfbar |
| [`extract_block_layout.py`](../../Simulation/g1_dex3_sim/extract_block_layout.py) | `detect` (Blobs, mit `--expect` gegen Grundwahrheit) und `extract` (Datensatz → `layout.json`) |
| [`g1_dex3_cfg.py`](../../Simulation/g1_dex3_sim/g1_dex3_cfg.py) | exportiert den Kamerateil weiter; für die Env ändert sich nichts |
| [`render_cotrain_dataset.py`](../../Simulation/g1_dex3_sim/render_cotrain_dataset.py) | `--layout`; `place_cubes` mit Vorrang Layout → Greifpunkt → zufällig |
| [`server_rl_run.sh`](../../Simulation/server_rl_run.sh) | `layout`, `layoutcheck`, `render` |

### Befehle

```bash
# lokal, ohne Container — Detektor auf den Referenzframes ansehen
cd Simulation/g1_dex3_sim
python3 extract_block_layout.py detect ../camera_reference/dataset_cam_left_high.png \
        --camera cam_left_high --debug-dir /tmp/dbg

# Server: Abnahme-Test (§7), dann Layout, dann Rendern
LAYOUTCHECK_FRAME=… LAYOUTCHECK_EXPECT=… ./Simulation/server_rl_run.sh layoutcheck
RENDER_EPISODES=60 LAYOUT_BIAS="dx dy" ./Simulation/server_rl_run.sh layout
RENDER_LAYOUT=/data/cotrain/layout.json RENDER_GRASP_WINDOW=600 \
        ./Simulation/server_rl_run.sh render
```

### Kennzahlen zum Wiedererkennen

| Größe | Wert |
|---|---|
| Auflösung / HFOV | 640 × 480 / 75° |
| `f_px` | 417,03 |
| Kamerapose `cam_left_high` | eye (0,0537 ; 0,025 ; 1,3239), 55° geneigt |
| Stereobasis | 4,7 cm, parallel blickend |
| Würfelebene `block_z_surface` | 0,915 (Tischoberkante 0,87) |
| Würfelkante | 5 cm ≙ 41,8 px auf 0,499 m |
| Auflösung auf der Ebene | 0,09–0,16 cm/px |

### Ausgabeformat `layout.json`

```jsonc
{
  "cameras": ["cam_left_high", "cam_right_high"],
  "bias_cm": [0.0, 0.0],
  "z_plane": 0.915,
  "episodes": {
    "0": {
      "frame": 0,
      "cubes": [[0.325, -0.058], [0.351, 0.120], [0.256, 0.158]],  // rot, grün, gelb
      "per_camera": { "cam_left_high": [ /* … */ ] },
      "camera_spread_cm": [2.04, 2.47, 1.30]
    }
  }
}
```

`cubes[i] == null` heißt „Würfel i nicht gefunden" — `place_cubes` fällt dann für genau
diesen Würfel auf den Greifpunkt bzw. den Zufall zurück, was im `cube_source`-Feld des
`render_manifest.json` protokolliert wird.
