# Würfellage aus den Realbildern

**Stand:** 2026-08-25 · aktuelles Verfahren: `pick_anchored_homography` Version 4
· Gierwinkel siehe [§7](#7-gierwinkel-um-die-eigene-z-achse)

Der Realdatensatz enthält Roboterzustände, Aktionen und Videos, aber keine Objektposen. Für einen
brauchbaren Replay muss der Sim-Würfel dort beginnen, wo er im Realbild lag. Andernfalls greift der
Roboter neben den Würfel und ein daraus erzeugter Co-Training-Datensatz koppelt Bild und Aktion
falsch.

Das aktuelle Verfahren lernt die Abbildung von den beiden Kopfkamerabildern nach Tisch-XY aus
realen Pick-Ereignissen. Es korrigiert weder FOV noch Brennweite aus der scheinbaren Würfelgröße
und
verschiebt keine Zielpose nachträglich zu einem Greifpunkt.

## 1. Aktueller Datenfluss

```text
40 Kalibrierungsepisoden
  ├─ Farbtracking beider Kopfkameras → Bewegungsbeginn je Würfel
  ├─ stabile Top-Face-Pixel strikt vor dem Bewegungsbeginn
  ├─ wenige observation.state-Zeilen direkt in Isaac setzen → Fingerkuppen-FK
  ├─ eindeutige Hand + Würfelfarbe → Pick-Anker Pixel ↔ Tisch-XY
  └─ 80/20-Split ganzer Episoden, Seed 17
       ├─ RANSAC-Homographie je Kamera auf Fit-Episoden
       └─ Qualitäts-Gates auf unberührten Holdout-Episoden

separat gewählte Ziel-Episoden
  ├─ stabile Würfelpixel vor ihrer ersten Bewegung
  ├─ Homographie je verfügbarer Kopfkamera
  ├─ Stereo-Median oder strenger Einzelkamera-Fallback
  └─ cube_poses.json v4
       └─ einmalig setzen → ausschließlich PhysX → replay_manifest.json v2
```

Die Auswahl ist bewusst getrennt:

- `REPLAY_CALIBRATION_NUM_EPISODES=40` wählt ab Episode 0 ausschließlich die
  Kalibrierungsmenge.
- `REPLAY_NUM_EPISODES`, `REPLAY_START_EPISODE` und `REPLAY_EPISODE_IDS` wählen ausschließlich
  die später zu rekonstruierenden und zu rendernden Ziel-Episoden.

Eine Ziel-Episode darf auch in den 40 Kalibrierungsepisoden enthalten sein. Der Holdout prüft die
Generalisierung der Homographie innerhalb der Kalibrierungsmenge; er ist kein Train/Test-Split des
späteren Replay-Datensatzes.

## 2. Anker aus Bild und sparsamer FK

### 2.1 Bewegungsbeginn und stationäre Bildmessung

`collect_replay_anchors.py` verfolgt rot, grün und gelb in beiden Kopfkameras bei halber
Auflösung. Die frühe Basisposition ist der Median der ersten zehn gültigen Messungen. Ein
Bewegungsbeginn wird erst akzeptiert, wenn der Farbmittelpunkt mindestens 8 px verschoben bleibt
und das für fünf aufeinanderfolgende Frames. Die Onsets der beiden Kameras dürfen höchstens
zwölf
Frames auseinanderliegen.

Die Oberseitenmessung verwendet ausschließlich Frames vor dem jeweiligen Kamera-Onset. Sie
benötigt mindestens fünf räumlich konsistente Messungen; Bildrandtreffer und Messungen mit mehr
als
12 px Abstand zum Median werden verworfen. Fehlt ein Onset, können frühe Frames im Bericht
erscheinen, aber nicht als Kalibrierungsanker verwendet werden.

### 2.2 Sparse Direct-State-FK

Rund um den visuellen Onset werden Schließintervalle im aufgezeichneten Gelenkzustand gesucht. Nur
deren Anfangs- und Endzustände sowie drei Zustände um das **Ende der Schließbewegung** (dessen
Frame ±2, nach oben auf den Onset begrenzt) werden direkt in Isaac gesetzt. Es findet kein
Action-Replay statt. Der Ankerzeitpunkt ist bewusst das Schließende und nicht der Onset: im
Abnahmelauf 2026-08-22 lagen dazwischen neun bis achtzehn Frames, in denen die Hand den Würfel
bereits anhob — alle zehn damals akzeptierten Anker waren dadurch nach vorne und oben versetzt.

Ein Anker wird nur akzeptiert, wenn:

- eine Hand mindestens 6 mm schließt;
- ihre Schließung mindestens 2 mm stärker als die andere Hand ist;
- die drei echten Fingerkuppen verfügbar sind;
- ihr Schwerpunkt in demselben Arbeitsraum liegt, den später auch die rekonstruierte Würfelpose
  erfüllen muss (x = 0,25–0,45 m, y = −0,25–0,25 m), und in z = 0,860–0,945 m, also am ruhenden
  Würfel (Tischplatte 0,870 m, Würfeloberseite 0,920 m, je eine halbe Kantenlänge Toleranz);
- beide Kameras einen ausreichend ähnlichen Bewegungsbeginn und stabile Vorher-Pixel liefern.

Der Anker besteht aus Episode, Farbe, Onset, Hand, den beiden Pixelmessungen und dem Median der
Fingerkuppenpositionen. Originalaktionen werden nur gehasht und nach der Verarbeitung erneut auf
Unverändertheit geprüft.

## 3. Homographie und Abnahme

Je Kopfkamera wird mit RANSAC eine projektive Abbildung von Pixel `(u, v)` nach env-lokalem
Tischpunkt `(x, y)` gelernt. Der RANSAC-Inlier-Schwellwert beträgt 2 cm. Der Split erfolgt auf
vollständigen Episoden: 80 % Fit, 20 % Holdout, deterministisch mit Seed 17.

`geometry_calibration.json` wird nur geschrieben, wenn alle Bedingungen gelten:

| Gate | Grenzwert |
|---|---:|
| Fit-Anker / Fit-Episoden | mindestens 24 / 8 |
| Holdout-Anker / Holdout-Episoden | mindestens 6 / 3 |
| Arbeitsraumabdeckung je Kamera | mindestens 12 cm in x und y |
| Ankerverteilung je Kamera | Nebenachsenstreuung ≥ 3 cm; größte Lücke je Hauptachse ≤ 40 % der Spannweite (ab 8 Ankern) |
| Holdout-Fehler je Kamera | Median ≤ 1,5 cm, p90 ≤ 3 cm |
| Differenz der Kameraschätzungen im Holdout | Median ≤ 2 cm, p90 ≤ 3 cm |

Das Ergebnis hat `version: 4`, `method: "pick_anchored_homography"` und `valid: true`. Ein
fehlgeschlagener Lauf schreibt weiterhin den vollständigen `calibration_report.json`, aber keine
verwendbare Kalibrierungsdatei. Eine eventuell vorhandene ungültig gewordene Ausgabedatei wird
entfernt.

Die aus einer achsenparallelen Bounding-Box geschätzte 5-cm-Würfelkante ist nur noch Diagnose. Der
fehlgeschlagene Lauf mit 3,68 cm zeigte, dass Perspektive, Gier und unvollständige Oberseiten diese
Skalenschätzung verzerren. Sie verändert deshalb ausdrücklich keine Intrinsics und erzeugt keine
FOV-Korrektur.

## 4. Anfangsposen der Ziel-Episoden

`replay-poses` verfolgt jede Farbe in jeder Ziel-Episode erneut. Es verwendet mindestens fünf
stabile Messungen vor dem jeweiligen Bewegungsbeginn und wendet die geprüfte Homographie der
jeweiligen Kamera an.

- Liefern beide Kameras eine Position, wird ihr Median verwendet. Mehr als 3 cm Differenz verwirft
  den Würfel.
- Liefert nur eine Kamera eine Position, sind mindestens fünf echte Top-Face-Detektionen nötig.
  Verwendet wird dann ausschließlich der transformierte Median dieser Top-Face-Pixel, nicht ein
  Full-Blob-Median.
- Das Ergebnis muss innerhalb x = 0,25–0,45 m und y = −0,25–0,25 m liegen.
- Fehlt einer der drei Würfel, erhält die Episode `status: "skipped"` und wird nicht gerendert.

`cube_poses.json` hat `version: 4` und `method: "stationary_top_face_homography"`. Die Höhe bleibt
fest beim Würfelmittelpunkt z = 0,915 m, die Orientierung vorerst bei Identität. Getrennte
Kameraschätzungen, verwendete Frames und Detektionsquelle stehen am Block. Die
`pick_anchor_diagnostics` stehen dagegen einmal im jeweiligen Episode-Eintrag. Auf Dokumentebene
stehen Quelldatensatz, Kalibrierungspfad und `calibration_sha256`.

Die Position stammt vollständig aus der Homographie. Es gibt keine 75/25-Mischung mit einem
Fingerkuppenschwerpunkt, keinen Greifpunkt-Fallback und keine Zufallsposition.

## 5. Physikbasierte Validierung

Der Renderer setzt Roboter und Würfel vor Frame 0. Danach gibt es keine weiteren Pose-Schreibungen,
kein Tracking und kein kinematisches Attach. Jede Bewegung eines Würfels stammt ausschließlich aus
PhysX unter den unveränderten Originalaktionen.

Vor dem Action-Replay werden die gesetzten Würfel in beiden Kopfkameras gegen ihre erwartete
Projektion geprüft. Die Abnahme verlangt sechs Messungen sowie höchstens 5 px Median und 10 px
p90.
Im Videomodus bleibt eine Abweichung als sichtbare Diagnose erlaubt; im Dataset-Modus wird die
Episode verworfen.

Während des Replays werden Würfel- und Fingerkuppenpositionen nur gelesen. Das Manifest Version 2
bewertet den erwarteten ersten Pick aus den Ankerdiagnosen. Erfolg bedeutet: Der erwartete Würfel
wird nach diesem Pick mindestens 2 cm angehoben und bleibt mindestens fünf aufeinanderfolgende
Frames über dieser Schwelle. Diese Metrik verändert die Simulation nicht.

Vor dem Rendern werden beide v4-Eingabeartefakte, ihre Methoden, ihr Quelldatensatz und der
Kalibrierungs-Hash in `cube_poses.json` geprüft. Overwrite umgeht diese Prüfung nicht. Im Manifest
Version 2 stehen `calibration_sha256` und `poses_sha256` auf oberster Ebene, nicht in den
Episode-Einträgen. Alte Renderausgaben werden nur fortgesetzt, wenn diese Felder und die
Griffvalidierung kompatibel sind. Andernfalls sind ein neues Ausgabeverzeichnis oder
`REPLAY_OVERWRITE=1` erforderlich.

## 6. Historische Ansätze und ihr Befund

Die folgenden Verfahren sind dokumentierte Fehlversuche oder Ablationen, nicht der aktuelle Pfad.

### 6.1 Globales Minimum der Fingeröffnung aus `scan.json`

Das Minimum liegt bei einem Pick-and-Place oft irgendwo im breiten geschlossenen Transportintervall
statt am Pick. Im Lauf vom 2026-08-17 lagen 48 von 116 erkannten Griffen hinter 60 % der Episode,
21 sogar hinter 80 %. Zusätzlich wurde z verworfen und nur ein Greifpunkt je Hand erzeugt. Dieser
Ansatz darf nicht mehr als Würfelposition verwendet werden.

### 6.2 Nur bis zum Griff rendern

`--stop-at-grasp` begrenzt falsche Frames nach dem Griff, rekonstruiert aber keine Anfangsposition.
Liegt der Würfel falsch, ist bereits die gesamte Anfahrt falsch. Der Schnitt kann für den alten
Co-Training-Renderer weiterhin eine Sicherheitsmaßnahme sein, gehört aber nicht zum neuen
physikbasierten Replay.

### 6.3 Blob-Schwerpunkt plus analytisches Pinhole-Modell

Der Schwerpunkt der sichtbaren Silhouette enthält Ober- und Seitenflächen und liegt nicht auf der
Projektion des Würfelmittelpunkts. Eine additive Bias-Korrektur ist nur lokal gültig. Außerdem
kann
ein in sich exakter `project`/`backproject`-Roundtrip einen gemeinsamen Fehler der realen und
simulierten Kamerapose nicht erkennen.

`camera_geometry.py` bleibt für Projektion, Debugging und Renderer-Abnahme verfügbar. Die
produktive Pixel-zu-Tisch-Abbildung der Version 4 kommt jedoch aus Holdout-geprüften Pick-Ankern.

### 6.4 Oberseiten-AABB als FOV-Kalibrierung und 75/25-Fusion

Die zwischenzeitliche Version schätzte die Kameraskala aus der größten achsenparallelen
Oberseiten-Ausdehnung und kombinierte erkannte Würfel pauschal zu 75 % mit einem Greifpunkt. Beide
Schritte sind entfernt: Die AABB ist bei Perspektive und Würfelgier kein zuverlässiges
Längennormal, und ein Greifpunkt ist ohne eindeutige zeitliche Farbzuordnung keine Startpose.

## 7. Gierwinkel um die eigene z-Achse

**Stand:** 2026-08-25 · gebaut und synthetisch abgenommen · auf Realbildern noch maskenlimitiert

Bis zum 2026-08-25 rekonstruierte keiner der beiden Pfade die Drehung. `place_cubes` schrieb ein
festes Identitäts-Quaternion, `_reset_idx` ebenso — jeder Würfel stand in jedem Sim-Lauf
achsparallel zur Tischkante, während er im Realdatensatz oft schräg liegt. §6.4 nannte die
Würfelgier bereits als Grund, warum die Oberseiten-AABB kein Längennormal ist; geschätzt wurde
sie trotzdem nie.

Das ist derselbe Fehler wie eine falsche Position, nur im Drehfreiheitsgrad. Das Co-Training-Paar
ist (Sim-Bild, **Real**-Aktion): die Realaktion richtete die Hand nach einem Würfel bei ψ aus, das
gerenderte Bild zeigt ihn bei 0°, und das Paar lehrt eine Zuordnung von Aussehen zu Handdrehung,
die es nicht gibt. Für den Griff kommt hinzu: über die Fläche ist ein 5-cm-Würfel 5,0 cm breit,
über die Diagonale 7,1 cm — steht er falsch, trifft die Hand eine Ecke statt einer Fläche.

### 7.1 Warum der naheliegende Weg nicht taugt

`replay_calibration.top_face_blob` berechnet seit jeher eine Min-Area-Box über die Deckfläche und
legt ihre Achse als `axis_uv` ab. Als Würfelorientierung ist sie aus zwei geometrischen Gründen
untauglich:

- **Sie misst im Bild**, also mitsamt der perspektivischen Verzerrung der Deckfläche.
- **Sie arbeitet auf einer binarisierten Kleinmaske.** Eine Min-Area-Box bevorzugt dort die
  Bildachsen, weil das Pixelraster selbst achsparallel ist.

> **Warnung zur Beweislage.** Beim ersten Anlauf hatte ich das an den PNGs in
> `runs/20260822/01/calibration_report/` „gemessen" (101 von 120 exakt 0,0°). Diese Dateien sind
> **annotierte Debug-Ausgaben**: 336 Pixel exakt (255,255,255) bilden einen Rechteckrahmen, 76 Pixel
> sind exakt (255,0,0). Der eingezeichnete Farbring ist damit der hellste gesättigte Bereich, und
> der Deckflächenschnitt griff den **Marker** statt der Würfeloberseite. Ob `axis_uv` auf echten
> Frames einrastet, ist folglich ungemessen — die beiden Punkte oben stehen auf Geometrie, nicht auf
> Daten. Wer hier Zahlen braucht, nimmt Videoframes über `detect`, nicht die Debug-PNGs.

### 7.2 Das Verfahren

`extract_block_layout.yaw_from_top_face`, zwei Entscheidungen:

1. **Erst zurückprojizieren, dann messen.** Die Deckflächenpixel werden auf `Z_CUBE_TOP` = 0,920
   geschnitten. Auf ihrer eigenen Ebene ist die Fläche wieder ein echtes Quadrat.
2. **Das 4. Winkelmoment statt einer Box.** Ein Quadrat ist 4-zählig, seine Richtung steckt in
   genau dieser Harmonischen: `Σ (dx + i·dy)^4` hat die Phase 4·ψ. Jedes Pixel geht mit stetigem
   Gewicht ein, es gibt kein Raster zum Einrasten.

Davor steht der **Deckflächenschnitt**, und der war bis zum 2026-08-25 der eigentliche Engpass. Nur
die Deckfläche liegt auf `Z_CUBE_TOP`; Seitenflächen-Pixel dorthin zurückzuprojizieren zieht sie zu
einem Schweif von der Kamera weg. Getrennt wurde über die Helligkeit — aber mit einer **festen
Quote**, dem 70. Perzentil, also „die hellsten 30 %". Bei 53,6° Blickhöhe macht die Deckfläche je
nach Gierwinkel 49–58 % der Silhouette aus; die Quote nahm damit gut die Hälfte davon. Vorhergesagte
Verkürzung linear 1,35, im Probelauf gemessen 1,42 (3,51 cm statt 5,0). Die Schwelle kommt jetzt aus
der Helligkeitsverteilung selbst (Otsu). Eine Zahl durch eine andere zu ersetzen wäre dieselbe
Wette gewesen — `topface` misst deshalb nach (§7.3).

Dazu zwei Prüfmerkmale, weil die Phase allein nicht sagt, ob die Punktwolke überhaupt ein Quadrat
ist. **Formprobe** = Diagonale/Kante der zurückprojizierten Fläche; ideal √2 = 1,41, nahe 1 heißt
„keine Kantenrichtung vorhanden". Ist sie kleiner als 1, ist die Phase um 45° umgeschlagen und
wird zurückgedreht. **Zwei-Kamera-Tor**: beide Kopfkameras müssen sich einig sein. Ihre Fehler
sind weitgehend unabhängig, weil die Verschmierung durch die Seitenflächen jeweils anderswohin
zeigt.

Wichtig: die Perzentil-Variante der Breite (5.–95.) **zerstört** die Unterscheidung — bei
gefüllten Flächen liegt das Verhältnis dann bei 1,08 statt 1,41, weil die Projektion quer zur
Diagonale dreieckig verteilt ist und quer zur Kante gleichverteilt. Es muss die volle Spannweite
sein.

### 7.3 Abnahme

`extract_block_layout.py selftest` (bzw. `server_rl_run.sh yawcheck`) rendert Würfel bekannter
Drehung durch dasselbe Kameramodell und misst zurück — über den vollen Weg inklusive Farbmaske,
Blobwahl und Deckflächenschnitt, nicht nur über die Formel. 18 Winkel × 5 Orte × 2 Kameras:

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

Diese Abnahme rendert ihre eigenen Bilder und ist deshalb von der Debug-PNG-Verwechslung in §7.1
unberührt — sie belegt aber auch nur den Schätzer, nicht das Verhalten auf Realbildern. Ohne die
beiden Tore lag derselbe Aufbau vor dem Schnitt-Fix bei Median 3,15° / p90 26,3°; das Tor kauft
Genauigkeit mit Ertrag, und das ist die richtige Richtung: ein geratener Winkel dreht den
Würfel unter einer Realaktion weg, die für eine andere Lage aufgenommen wurde.

Die Abnahme gegen den **echten Renderer** ist `detect --expect-yaw` mit `cubes_yaw_deg` aus
`render_manifest.json` — dieselbe Rolle, die `--expect` für die Position spielt. Der `selftest`
teilt sich mit dem Schätzer die Kameraannahme und kann sie deshalb nicht prüfen.

### 7.4 Der Probelauf und was er ergab

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
Fehler auf zwei Stellen vorher (siehe §7.2: erwartet 1,35, gemessen 1,42). Die Schwelle wird deshalb
seit dem 2026-08-25 gemessen statt gesetzt.

### 7.5 Der Deckflächenschnitt, auf echten Frames vermessen

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

### 7.6 Probelauf statt Vollauslauf

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
| `replay_anchors.json` | v4-Anker, Ablehnungsgründe, Action-Hashes und Bilddiagnosen |
| `geometry_calibration.json` | gültige v4-Homographien, Fit/Holdout und Qualitätswerte |
| `calibration_report.json` | vollständiger Bericht auch bei fehlgeschlagenem Gate |
| `cube_poses.json` | v4-Anfangsposen der separat ausgewählten Ziel-Episoden |
| `replay_manifest.json` | v2-Render-, Projektions- und Griffdiagnose |

```bash
REPLAY_CALIBRATION_NUM_EPISODES=40 \
REPLAY_CALIBRATION_HOLDOUT_RATIO=0.2 \
REPLAY_CALIBRATION_SEED=17 \
./Simulation/server_rl_run.sh replay-calibrate

REPLAY_EPISODE_IDS="0 12 41" REPLAY_OVERWRITE=1 \
./Simulation/server_rl_run.sh replay-poses

DR_ENABLED=0 REPLAY_EPISODE_IDS="0 12 41" REPLAY_OVERWRITE=1 \
./Simulation/server_rl_run.sh replay-render
```

Die vollständige Bedienung und alle Umgebungsvariablen stehen in
[replay-videos-aus-realdaten.md](replay-videos-aus-realdaten.md).
