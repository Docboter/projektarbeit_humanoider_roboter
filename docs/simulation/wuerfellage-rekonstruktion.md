# Würfellage aus den Realbildern — Verfahren, Koordinaten, offene Punkte

> **TL;DR:** Beschreibt Pfad A — das aktuelle Verfahren zur Würfellage-Rekonstruktion aus
> Realbildern (Farbsegmentierung → Kamerastrahl → Schnitt mit der Würfelebene) — sowie die beiden
> gescheiterten Vorgängerversuche. Übergabedokument für den Co-Training-Renderer; Code-Audit und
> Abnahmelauf-Befund stehen in wuerfellage-rekonstruktion-bewertung.md bzw.
> wuerfellage-rekonstruktion-lauf35-befund.md.

**Stand:** 2026-10-11 (Verfahren: 2026-08-17, Gierwinkel: 2026-08-25) · Position: im Sim-Render auf
rund 1 cm bestätigt (`layoutcheck`, 2026-08-22, [Befund](wuerfellage-rekonstruktion-lauf35-befund.md) §8);
dass die **reale** Kopfkamera derselben Pose folgt, ist weiterhin nicht belegt · Vorgeschichte: [co-training.md](../training/co-training.md) §3.0/§3.2a

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
(Kameramodell). Beide brauchen **kein Isaac Lab und keine GPU** — das ist Absicht: nur so
lässt sich das Kameramodell auch außerhalb des Containers prüfen.

```text
Realvideo Episode n, Frame 0
   │
   ├─ (a) HSV-Segmentierung rot / grün / gelb           → binäre Masken
   ├─ (b) größte zusammenhängende Fläche je Farbe       → Blob
   ├─ (c) Schwerpunkt des Blobs                         → Pixel (u, v)
   ├─ (d) Pixel → Kamerastrahl → Schnitt mit z = 0.895  → (x, y) env-lokal
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
[`g1_dex3_blockstack_env.py`](../../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py). Die
RGB→HSV-Umrechnung ist in numpy ausgeschrieben statt aus OpenCV geholt — das Werkzeug soll auch in
einem Container ohne `cv2` laufen.

### (b)/(c) Blob und Schwerpunkt

Pro Maske die größte zusammenhängende Fläche (4-Nachbarschaft), Mindestfläche 120 px; deren
Pixelschwerpunkt ist `(u, v)`. **Bekannter Bias:** der Blob ist die Silhouette eines schräg von oben
gesehenen Würfels (Deckfläche **plus** zugewandte Seitenfläche), sein Schwerpunkt liegt daher zur
Kamera hin verschoben — siehe (e).

### (d) Rückprojektion

Pixel → Kamerastrahl → Schnitt mit der Würfelebene z = 0,895 (`Z_CUBE_CENTER`, Mittelpunkt eines auf
dem 0,87 m hohen Tisch ruhenden 5-cm-Würfels), im Modul
[`camera_geometry.py`](../../Simulation/g1_dex3_sim/camera_geometry.py). Bis zum 2026-08-22 stand
hier 0,915; die Würfel ruhen aber bei 0,895, der Rest-Bias in x war genau diese falsche Ebene
([Befund](wuerfellage-rekonstruktion-lauf35-befund.md) §8).

### (e) Bias-Korrektur

Additive Verschiebung `(dx, dy)` in Metern (`--bias`, bzw. `LAYOUT_BIAS="dx dy"`), **gemessen**, nicht
geschätzt: `detect --expect` auf einem gerenderten Sim-Bild mit bekannten Würfelpositionen liefert das
mittlere Residuum. Die Streuung bleibt.

### (f) Zwei Kameras

`cam_left_high` und `cam_right_high` liefern je eine Schätzung; gespeichert wird das Mittel, die
Differenz als `camera_spread_cm` im `layout.json`. Die Stereobasis beträgt nur 4,7 cm bei ~0,50 m
Motivabstand: ein Fehler im **Kameramodell** verschiebt beide Strahlen fast gleich und ist in dieser
Differenz unsichtbar — sie prüft die **Detektion**, nicht die Pose.

### (g) Gierwinkel um die eigene z-Achse

Bis zum 2026-08-25 rekonstruierte das Verfahren die Drehung nicht: `place_cubes` und `_reset_idx`
schrieben ein Identitäts-Quaternion, jeder Würfel stand achsparallel. Im Realdatensatz liegen sie
schräg (§6). Das ist derselbe Fehler wie eine falsche Position, nur im Drehfreiheitsgrad: die
Realaktion richtete die Hand nach einem Würfel bei ψ aus, das Bild zeigt ihn bei 0°.

`extract_block_layout.yaw_from_top_face` trifft zwei Entscheidungen:

1. **Erst zurückprojizieren, dann messen.** Die Deckflächenpixel werden auf `Z_CUBE_TOP` = 0,920
   geschnitten; dort ist die Fläche wieder ein Quadrat. Die naheliegende Min-Area-Box im Bild
   (`top_face_blob`) misst mitsamt Perspektive und rastet auf der Kleinmaske zu Bildachsen ein.
2. **Das 4. Winkelmoment statt einer Box.** `Σ (dx + i·dy)^4` hat die Phase 4·ψ; jedes Pixel geht
   stetig ein, es gibt kein Raster zum Einrasten.

Davor steht der **Deckflächenschnitt** über die Helligkeit (Schwelle seit 2026-08-25 per Otsu
gemessen statt fix das 70. Perzentil, siehe §4/§5). Dazu zwei Prüfmerkmale: die **Formprobe**
(Diagonale/Kante der zurückprojizierten Fläche, ideal √2 = 1,41; < 1 heißt Phase um 45° umgeschlagen
und wird zurückgedreht) und das **Zwei-Kamera-Tor** (beide Kopfkameras müssen sich einig sein). Die
Breite muss die volle Spannweite sein — die 5.–95.-Perzentil-Variante zerstört die Unterscheidung
(Verhältnis 1,08 statt 1,41).

Abnahme: `extract_block_layout.py selftest` (bzw. `server_rl_run.sh yawcheck`) rendert Würfel
bekannter Drehung durch dasselbe Kameramodell und misst über den vollen Weg (Farbmaske, Blobwahl,
Deckflächenschnitt) zurück — 18 Winkel × 5 Orte × 2 Kameras:

| Maskenrauschen | Tor behält | Fehler Median | p90 | Ausreißer > 20° |
|---|---|---|---|---|
| 0,00 | 99 % | 0,08° | 0,32° | 0/89 |
| 0,01 | 99 % | 0,08° | 0,27° | 0/89 |
| 0,02 | 99 % | 0,07° | 0,27° | 0/89 |
| 0,03 | 100 % | 0,07° | 0,28° | 0/90 |
| 0,05 | 100 % | 0,09° | 0,33° | 0/90 |

Die Schranke steht bei p90 ≤ 1,0° und ≤ 1 % Ausreißer je Rauschstufe. Sie war bis zum 2026-08-25 bei
5° und 2 % — nachgezogen, weil ein Schwellwert, den der Ist-Zustand um das Dreifache unterbietet,
keine Regression mehr fängt.

Der synthetische Würfel wird **Lambert-schattiert mit schräger Lichtquelle**, nicht mit zwei festen
Helligkeiten. Das ist keine Kosmetik: bei senkrechtem Licht steht die Deckfläche so weit über jeder
Seitenfläche, dass sie jede Schwelle trennt — die Abnahme wäre ein Gummistempel. Schräg beleuchtet
liegt die hellste Seitenfläche bei 0,64 gegen 0,91, und genau dort entscheidet sich der Schnitt.
Unter dieser Beleuchtung fällt die alte feste Quote auf p90 **36,7°**, die gemessene Schwelle bleibt
bei 0,20°.

Diese Abnahme rendert ihre eigenen Bilder und ist deshalb von der Debug-PNG-Verwechslung
([zurückgezogene Erstbegründung, Lauf 52](../ergebnisse/diagnose-chronik.md#lauf-52-der-gierwinkel-fehlte-in-jedem-lauf))
unberührt — sie belegt aber auch nur den Schätzer, nicht das Verhalten auf Realbildern. Ohne die
beiden Tore lag derselbe Aufbau vor dem Schnitt-Fix bei Median 3,15° / p90 26,3°; das Tor kauft
Genauigkeit mit Ertrag, und das ist die richtige Richtung: ein geratener Winkel dreht den
Würfel unter einer Realaktion weg, die für eine andere Lage aufgenommen wurde.

Die Abnahme gegen den **echten Renderer** ist `detect --expect-yaw` mit `cubes_yaw_deg` aus
`render_manifest.json` — dieselbe Rolle, die `--expect` für die Position spielt. Der `selftest`
teilt sich mit dem Schätzer die Kameraannahme und kann sie deshalb nicht prüfen.

Ein Code-Audit des Parallelpfads B (`pick_anchored_homography`) steht in
[wuerfellage-rekonstruktion-bewertung.md](wuerfellage-rekonstruktion-bewertung.md).

## 4. Der Probelauf und was er ergab

Probelauf vom 2026-08-25 über die echten Videos der Episoden 0, 1, 2, 8 und 12 (`layout` →
`layoutreport`): **1 von 15 Würfeln** passiert das Tor, dieser eine bei **30,0°** — eine echte
Schräglage, kein Rundungsrest. Alle drei Würfel werden in allen fünf Episoden gefunden, die Kameras
sind sich in der *Position* auf 1,75 cm (Median) einig. Es scheiterte ausschließlich am Winkel.

`layoutreport` benennt, woran:

| Größe | gemessen | Soll |
|---|---|---|
| `fill` (Füllgrad der Bounding-Box) | 0,69 | ~0,7 — mehr kann ein Sechseck nicht |
| `top_px` | 494 | groß genug |
| `top_width_cm` | **3,51** | 5,0 |
| `top_squareness` | **1,17** | 1,41 |

und die Aufschlüsselung: **13 von 15 an der Formprobe**, 1 an der Uneinigkeit, 0 mangels Messung.

**Die Farbmaske war es also nicht** — 0,69 ist für eine sechseckige Silhouette in einer rechteckigen
Bounding-Box praktisch der Bestwert. Es war der Deckflächenschnitt, und die Geometrie sagt den
Fehler auf zwei Stellen vorher (erwartet 1,35, gemessen 1,42 — hergeleitet in
[Lauf 52 der Diagnose-Chronik](../ergebnisse/diagnose-chronik.md#lauf-52-der-gierwinkel-fehlte-in-jedem-lauf)). Die Schwelle wird deshalb
seit dem 2026-08-25 gemessen statt gesetzt.

## 5. Der Deckflächenschnitt, auf echten Frames vermessen

`topface` (`server_rl_run.sh topface`) stellt die Schwellenregeln gegeneinander. Die Würfeloberseite
ist ein 5-cm-Quadrat — „welche Regel trifft sie am besten" ist damit eine Messung und keine Meinung.
Lauf vom 2026-08-25, dieselben fünf Episoden:

| Regel | Breite | Formprobe | \|L−R\| | Deckfläche |
|---|---|---|---|---|
| gemessen (Otsu) | **4,94 cm** | 1,21 | **2,8°** | 903 px |
| q45 | 4,95 cm | 1,17 | 3,8° | 874 px |
| q55 | 3,84 cm | 1,19 | 3,2° | 701 px |
| q70 (alt) | 3,51 cm | 1,17 | 10,2° | 494 px |

Die gemessene Schwelle trifft die Kantenlänge auf 1 % und drückt die Uneinigkeit beider Kameras von
10,2° auf 2,8°. Zwei unabhängig aufgestellte Kameras, die sich auf 2,8° einig sind, messen mit hoher
Wahrscheinlichkeit denselben, richtigen Winkel.

**Die Formprobe blieb trotzdem bei 1,21** und warf damit 14 der 15 Würfel weg. Die Eichtabelle, die
`topface` mitdruckt, entscheidet den Fall — sie senkt die Schwelle schrittweise und zeigt, was mit
der Uneinigkeit der zusätzlich durchgelassenen Würfel passiert:

| Schwelle | behalten | \|L−R\| Median | p90 | max |
|---|---|---|---|---|
| 1,25 | 1/15 | 3,7° | 3,7° | 3,7° |
| 1,20 | 4/15 | 1,1° | 3,0° | 3,7° |
| 1,10 | 6/15 | 1,8° | 3,3° | 3,7° |
| 1,00 | **9/15** | 1,6° | 3,0° | **3,7°** |

Die Uneinigkeit steigt beim Senken **nicht** — ihr Maximum bleibt über alle Stufen exakt 3,7°. Die
Formprobe trennt auf echten Frames also nichts; sie hat nur 8 von 9 brauchbaren Würfeln weggeworfen.
Der Grund: 1,25 stammt von scharfkantigen synthetischen Würfeln (dort 1,37–1,38), echte Klötzchen
liegen mit gerundeten Kanten bei 1,21 (p10 1,06), dazu Bewegungsunschärfe und Maskenrand.

Synthetisch bestätigt sich dasselbe: bei intakter Deckfläche kostet die Formprobe fast nichts und
bringt nichts — mit 1,25 überleben 345 von 375 bei p90 0,20°, ganz ohne sie 373 bei p90 0,27°, beide
Male **null** Ausreißer über 20°. Sie war gegen den 45°-Umschlag gebaut, und den erzeugten zerfetzte
Deckflächen, die es mit der gemessenen Helligkeitsschwelle nicht mehr gibt.

**Vorgabe ist deshalb `--min-squareness 1.0`, also aus.** Nach der Umschlagkorrektur ist das
Verhältnis konstruktionsbedingt ≥ 1; die Zahl bleibt als Diagnose stehen und als Stellschraube für
den, der einer anderen Optik misstraut. Das einzige Tor ist die Einigkeit beider Kameras — und die
teilt die Würfel sauber: neun mit ≤ 3,7°, sechs mit > 8°, dazwischen nichts. Auf diesen fünf
Episoden steigt der Ertrag damit von 1/15 auf **9/15**.

Solange `cubes_yaw_deg` `null` ist, verhält sich der Renderer für diesen Würfel wie bisher. `null`
heißt „nicht belastbar gemessen", **nicht** „liegt gerade".

Unabhängige Gegenprobe: `grasp_anchor` berichtet den **Greifachsen-Winkel** aus der FK (Daumen →
Mitte von Zeige- und Mittelfinger, mod 90°) neben dem Bildwinkel. Wer greift, legt die Greifachse
quer zu einer Fläche. Er wird bewusst **nicht** als Ersatz für einen verworfenen Bildwinkel
eingesetzt — das wäre genau die stille Rückfallebene, die `place_cubes` aus gutem Grund verloren
hat. Erst wenn beide Zahlen über viele Episoden zusammenfallen, ist die Annahme belegt.

## 6. Der Vollauslauf: die Würfel liegen beliebig

60 Episoden, 2026-08-25. Alle drei Würfel in allen 60 Episoden gefunden, Kameras in der Position auf
1,17 cm (Median) einig, **110 von 180 Würfeln (61 %)** mit belastbarem Gierwinkel.

Die Verteilung ist das eigentliche Ergebnis:

| | gemessen | Gleichverteilung auf 0…45° |
|---|---|---|
| Median | **22,8°** | 22,5° |
| p90 | **40,7°** | 40,5° |
| max | **44,7°** | 45° |

Die Würfel liegen also nicht „meist gerade mit gelegentlichen Ausreißern", sondern in **beliebiger
Drehung**. Die Sim stellte jeden einzelnen auf 0° — bei einer Gleichverteilung der Punkt mit dem
größten Erwartungsfehler.

Ein verrauschter Schätzer sähe mod 90° allerdings ebenfalls gleichverteilt aus. Zwei Prüfungen
schließen das aus:

- **Die Kameras sind sich einig.** Die 110 Würfel haben das Einigkeitstor passiert; im Probelauf lag
  ihre Uneinigkeit bei ≤ 3,7°. Zwei unabhängig aufgestellte Kameras stimmen bei Rauschen nicht
  überein — Gleichverteilung aus Rauschen wäre je Kamera eine andere.
- **Das Tor wählt kaum nach Winkel aus.** Synthetisch gemessen liegt der Ertrag bei 0–10°
  Schräglage bei 97,6 % gegen 100 % darüber (Korrelation +0,32). Zu schwach, um aus einer gehäuften
  Verteilung eine gleichmäßige zu machen. Auf echten Daten prüft das `layoutreport` nach: es druckt
  die Verteilung der durchgelassenen Würfel **neben** der aller Einzelmessungen. Decken sie sich,
  formt das Tor nichts.

Folge für die Env: `block_yaw_range_deg` sollte für Eval, Replay und RL auf `(0, 90)` stehen, nicht
auf `(0, 0)`. Der Default bleibt vorerst `(0, 0)`, weil die Umstellung die Vergleichbarkeit mit den
Läufen 08–52 bricht — das ist eine Entscheidung über den Versuchsaufbau, keine Fehlerbehebung.

Für den Co-Training-Render spielt das keine Rolle: dort kommt der Winkel aus `layout.json`.
Die 39 % ohne Winkel werden mit 0° gerendert und tragen damit den alten Fehler weiter. Sie
auszuschließen wäre teuer (alle drei Würfel gemessen: ~0,61³ ≈ 23 % der Episoden); die zielgenauere
Variante wäre, nur den GEGRIFFENEN Würfel zu verlangen — den kennt `grasp_anchor`.

## 7. Probelauf statt Vollauslauf

Ein `layout`-Lauf über 60 Episoden dekodiert für den Bewegungsbeginn jedes Video einmal ganz —
rund 4 s je Video und Kamera, also etwa acht Minuten allein dafür. Für die Frage „kommt ein
Gierwinkel heraus?" reicht eine Handvoll Episoden in einer **eigenen** Ausgabedatei:

```bash
LAYOUT_OUT=/data/cotrain/layout_yawprobe.json \
LAYOUT_EPISODE_IDS="0 1 2 8 12" \
LAYOUT_NO_MOTION_ONSET=1 \
./Simulation/server_rl_run.sh layout
```

Zwei Fallen lagen dabei im Weg, beide seit 2026-08-25 entschärft und durch
`test_layout_file_merge.py` festgehalten:

- **`--overwrite` fing mit einer leeren Datei an.** Zusammen mit `--episode-ids` blieben danach
  nur die neu gerechneten Episoden übrig, die übrigen 55 waren weg — samt ihrer teuer
  gemessenen Bewegungsbeginne, und die Datei sah hinterher gültig aus. `--overwrite` rechnet
  jetzt nur die gewählten Episoden neu; für den bewussten Neuanfang gibt es `--fresh`.
- **`--no-motion-onset` löschte einen vorhandenen Bewegungsbeginn.** Es heißt „nicht messen",
  nicht „löschen" — ein schneller Teillauf machte die Datei sonst stillschweigend
  renderuntauglich. Vorhandene Werte bleiben jetzt stehen.

Zusätzlich lehnt `extract` es ab, Einträge zu mischen, die mit anderem `--bias`, anderer
Würfelebene oder anderen Kameras entstanden sind: in der Datei wären sie nicht mehr
unterscheidbar.

## 8. Artefakte und Befehle

| Artefakt | Inhalt |
|---|---|
| `layout.json` | je Episode Position (x, y), `camera_spread_cm` und `cubes_yaw_deg` (`null` = nicht belastbar gemessen); Standardpfad `/data/cotrain/layout.json` |
| `layout_debug/` | Debug-PNGs der Detektion (`LAYOUT_DEBUG_DIR`) — annotiert, **keine** Messgrundlage für den Winkel |
| `render_manifest.json` | Ausgabe von `render`, enthält `cubes_yaw_deg` für `detect --expect-yaw` |

```bash
./Simulation/server_rl_run.sh layout          # layout.json aus den Realvideos
./Simulation/server_rl_run.sh layoutreport    # woran der Gierwinkel scheitert
./Simulation/server_rl_run.sh topface         # Deckflächenschnitt vermessen
./Simulation/server_rl_run.sh yawcheck        # Schätzer gegen synthetische Würfel
./Simulation/server_rl_run.sh layoutcheck     # Kameramodell gegen ein Renderbild
./Simulation/server_rl_run.sh render          # Co-Training-Datensatz mit layout.json
```

Der frühere Parallelpfad B (`replay-prepare` → `replay-calibrate` → `replay-poses` →
`replay-render`, Artefakte `replay_anchors.json`, `geometry_calibration.json`, `cube_poses.json`)
wurde nach dem Abnahmelauf ([Befund](wuerfellage-rekonstruktion-lauf35-befund.md)) aufgegeben; sein
Protokoll steht in [replay-videos-aus-realdaten.md](replay-videos-aus-realdaten.md).
