# Abnahmelauf v4 (`replay-calibrate`, 40 Episoden) — Befund

> **TL;DR:** Befund des Abnahmelaufs von `pick_anchored_homography` (Pfad B, v4): Die
> Kalibrierung schlägt zweimal fehl, weil der Fingerkuppen-Anker die Würfelposition um 4–11 cm
> verfehlt (offene statt greifende Hand). §8 bestätigt stattdessen das Pinhole-Modell aus Pfad A
> auf rund 1 cm Genauigkeit und empfiehlt, `pick_anchored_homography` aufzugeben. Setzt
> wuerfellage-rekonstruktion-bewertung.md voraus, keine eigene Verfahrensbeschreibung.

**Lauf:** 2026-08-22, `Simulation/runs/20260822/01`, Commit `d73c6aa`, Episoden 0–39.
**Ergebnis:** Kalibrierung abgelehnt, `geometry_calibration.json` **nicht** geschrieben.
Das Fail-safe hat funktioniert — der Lauf hat keine schlechte Kalibrierung durchgelassen.

## 1. Was gemeldet wurde

```text
10 Anker aus 40 Episoden
QUALITY-GATE: nur 8 statt 24 Fit-Anker
QUALITY-GATE: nur 2 statt sechs Holdout-Anker
QUALITY-GATE: weniger als drei Holdout-Episoden
QUALITY-GATE: cam_left_high:  Holdout Median/P90 0.034/0.057 m
QUALITY-GATE: cam_right_high: Holdout Median/P90 0.077/0.091 m
QUALITY-GATE: Kameradifferenz Median/P90 0.048/0.083 m
```

Die drei Fehlerbeträge sind **Folge**, nicht Ursache: mit 8 Fit-Ankern ist eine
8-DOF-Homographie praktisch exakt bestimmt (6 Inlier je Kamera), und 2 Holdout-Anker
messen nichts. Die eigentlichen Befunde liegen davor.

## 2. Ursache 1 — die Bewegungsverfolgung mittelt über die ganze Farbmaske

`track_colors` bildet den Schwerpunkt **aller** Pixel der Farbmaske
([replay_calibration.py:131](../../Simulation/g1_dex3_sim/replay_calibration.py#L131)),
während der Messpfad (`top_face_blob` → `largest_blob`) korrekt den **größten Blob**
nimmt. Die Maske enthält aber nur zu 47–64 % den Würfel:

| Farbe | Maskenpixel | Komponenten | größte | Abstand Gesamt- zu Blob-Schwerpunkt |
|---|---:|---:|---:|---:|
| rot | 1314 | 4 | 47 % | 7 px |
| gruen | 631 | 27 | 64 % | 4 px |
| gelb | 1369 | 16 | 64 % | **14 px** |

(Episode 0, `cam_left_high`, jeweils der im Bericht abgelegte Referenzframe.)

Die Onset-Schwelle ist 8 px ([replay_calibration.py:137](../../Simulation/g1_dex3_sim/replay_calibration.py#L137)).
Bei Gelb überschreitet allein die Kontamination diese Schwelle. Ergebnis:

- **Gelb: Median-Onset Frame 18, Minimum 10** (Frame 10 ist der frühestmögliche Wert
  überhaupt). 33 von 40 Gelb-Onsets liegen bei ≤ Frame 30 — der Würfel „bewegt sich",
  bevor der Roboter ihn berührt. **Gelb liefert 0 von 40 Ankern.**
- Die beiden Kopfkameras widersprechen sich massiv: |Δ Onset| **Median 20 Frames,
  p90 354, Maximum 674** bei einem Gate von 12. **67 von 120** (Episode × Farbe) fallen
  genau hier durch.
- Die restlichen **43 von 120** fallen unter „keine eindeutige physische Handschließung" —
  überwiegend, weil der Falsch-Onset am Episodenanfang liegt und die Hand dort *öffnet*
  statt zu schließen (`finger_closure_m` negativ, z. B. −6,6 mm und −12,9 mm).

Verteilung: rot 9 Anker, gruen 1, gelb 0.

## 3. Ursache 2 — der Anker misst die Hand, nachdem sie den Würfel angehoben hat

`pick_anchor` nimmt die Fingerkuppen bei **onset−2 / onset / onset+2**
([collect_replay_anchors.py:131](../../Simulation/g1_dex3_sim/collect_replay_anchors.py#L131)).
Die zugehörige Schließbewegung endet laut den akzeptierten Ankern aber **9–18 Frames
früher** (z. B. Episode 0: Schließung 90–99, Onset 108). Gemessen wird also 0,25–0,4 s
nach dem Griff — mitten im Anheben.

Die Anker-z-Werte belegen das modellfrei. Würfelmittelpunkt 0,915 m, Oberseite 0,940 m
(Tisch 0,890 m, aus der Kalibrierung selbst):

| Episode | Farbe | Anker-z | über der Würfeloberseite |
|---:|---|---:|---:|
| 0 | rot | 0,975 | +3,5 cm |
| 1 | rot | 0,985 | +4,5 cm |
| 7 | rot | 0,982 | +4,2 cm |
| 9 | gruen | 0,942 | +0,2 cm |
| 11 | rot | 1,009 | +6,9 cm |
| 13 | rot | 1,039 | +9,9 cm |
| 15 | rot | 1,000 | +6,0 cm |
| 19 | rot | 0,997 | +5,7 cm |
| 30 | rot | 0,990 | +5,0 cm |
| 32 | rot | 1,025 | +8,5 cm |

Neun von zehn Ankern liegen 3,5–9,9 cm **über** der Oberseite eines Würfels, der dort
noch ruhen soll. Eine Hand, deren drei Fingerkuppen im Mittel 5 cm über dem Würfel
schweben, hält diesen Würfel nicht an seiner Ausgangsposition.

**Gegenprobe mit dem unabhängigen Pinhole-Modell** ([camera_geometry.py](../../Simulation/g1_dex3_sim/camera_geometry.py),
Pfad A, Isaac-frei): dieselben stabilen Top-Face-Pixel auf z = 0,915 zurückprojiziert
ergeben für alle 120 Messungen x ∈ [0,229 … 0,406] (Median 0,325) und
y ∈ [−0,072 … 0,176] — also mitten im Spawnband der Env (x 0,30–0,40). Die beiden
Kameras stimmen dabei auf **1,7 cm** überein. Die Fingerkuppen-Anker liegen dagegen
**systematisch 11,5 cm weiter vorne** (Δx Median −0,115 m links, −0,119 m rechts;
Δy nur +0,012 m).

Die Hypothese „die reale Tischebene liegt anders als 0,915" ist damit ausgeschlossen:
eine Rasterung über z = 0,80 … 1,30 m findet **keine** Ebenenhöhe, bei der die
Würfelpixel auf die Fingerkuppen fallen — das Optimum läuft an den unteren Rand und
behält 1–9 cm Restfehler. Der Versatz ist nach vorne **und** nach oben, also eine
Bewegung, keine Ebene.

> Einschränkung: Das Pinhole-Modell ist laut eigenem Dateikopf rekonstruiert, nicht
> kalibriert; sein x hängt direkt am geschätzten Nickwinkel. Ein Teil der 11,5 cm kann
> dort sitzen — Episode 9 hat korrekte Höhe und trotzdem 11 cm Versatz. Die z-Evidenz
> aus der Tabelle oben hängt aber nur an FK und Tischhöhe und bleibt davon unberührt.
> `server_rl_run.sh layoutcheck` entscheidet das und kostet Minuten.

## 4. Zwei Gates, die den Fehler nicht gefangen haben

- **Arbeitsraum-Gate des Ankers** ist x 0,20–0,50, y ±0,30, **z 0,82–1,05**
  ([collect_replay_anchors.py:136](../../Simulation/g1_dex3_sim/collect_replay_anchors.py#L136)).
  Das 23-cm-z-Fenster lässt jeden der zehn falschen Anker durch. Ein Fenster von
  0,90–0,97 hätte 9 von 10 verworfen — der Lauf wäre mit „keine Anker" gescheitert
  statt mit einer scheinbar knappen Homographie. Nebenbei: die Anker erreichen x = 0,497,
  außerhalb der Zielgrenzen (0,25–0,45), die dieselbe Pipeline für Würfelposen erzwingt.
- **Abdeckungs-Gate** prüft `np.ptp` ≥ 12 cm je Achse
  ([replay_calibration.py:343](../../Simulation/g1_dex3_sim/replay_calibration.py#L343)).
  Die zehn Anker liegen in zwei Klumpen (y ≈ −0,06…−0,12 bei rechter Hand, +0,15…+0,22
  bei linker; dazwischen nichts). Spannweite besteht, Abdeckung nicht — genau der
  entartete Fall, den das Gate verhindern sollte.

## 5. Empfohlene Reihenfolge

> **Stand 2026-08-22:** Punkt 1–4 sind umgesetzt (`track_colors`, `pick_anchor`,
> gemeinsame Arbeitsraum-Konstanten in `replay_calibration.py`, `coverage_defects`).
> Lokal verifiziert sind nur die beiden Gates — gegen die aufgezeichneten Anker dieses
> Laufs meldet `coverage_defects` „Hauptachse geklumpt (66 %)", und das neue
> Fingerkuppenfenster verwirft 9 der 10 Anker. Die Wirkung von Punkt 1 und 2 auf die
> Ankerausbeute lässt sich nur auf dem Server messen: `replay-calibrate` erneut fahren.

1. **`track_colors` auf den größten Blob umstellen** (Konsistenz mit dem Messpfad).
   Billigste Änderung mit der größten Wirkung: sie adressiert die 67 Onset-Konflikte
   und die 33 Gelb-Fehlauslösungen zugleich.
2. **Fingerkuppen am Schließereignis abgreifen** statt am Onset — `event["end_frame"]`
   statt `onset ± 2`. Behebt den +11,5 cm/+7,5 cm-Versatz an der Wurzel.
3. **z-Fenster des Ankers auf ±3–4 cm um die Würfelmitte** verengen, damit ein künftiger
   Rückfall laut scheitert statt leise zu kalibrieren.
4. **Abdeckung über Streuung/Kondition** statt Spannweite prüfen.
5. Gelb-HSV-Fenster (`h` 35–70°, `s` ≥ 0,35, `v` ≥ 0,28) gegen die Fehlpixel schärfen
   oder die Suche auf die Tischregion begrenzen.
6. Danach `replay-calibrate` wiederholen. Erst wenn die Anker stehen, sind Holdout-Zahlen
   und Kameradifferenz überhaupt interpretierbar.

## 6. Was der Lauf nicht sagt

Homographie-Qualität, Wrist-Gate, Dataset-Modus und Renderpfad sind ungeprüft — die
Pipeline ist nie über Schritt 1 hinausgekommen. Die Bewertung in
[wuerfellage-rekonstruktion-bewertung.md](wuerfellage-rekonstruktion-bewertung.md)
bleibt ansonsten unverändert gültig; §5.2 (Anker ≠ Würfelmittelpunkt, 23-cm-z-Fenster)
ist durch diesen Lauf von einer Vermutung zu einem gemessenen Befund geworden.

---

## 7. Zweiter Lauf nach den Korrekturen (`Simulation/runs/20260822/02`)

`replay-calibrate`, wieder 40 Episoden, mit den Korrekturen 1–4. Ergebnis: **3 Anker statt 10**,
Kalibrierung erneut abgelehnt.

### 7.1 Korrektur 1 hat gewirkt

Die Bewegungsverfolgung ist repariert. Gegenüber Lauf 01:

| Farbe | Onset-Median | Kameradifferenz Median | Fälle mit \|Δ\| > 12 |
|---|---:|---:|---:|
| rot | 120 → 118 | 1 → 2 | 14 → 11 |
| gruen | 216 → **588** | 50 → **2** | 28 → 11 |
| gelb | 18 → **274** | 21 → **4** | 25 → 16 |

Gelb feuerte vorher im Median bei Frame 18, also bevor der Roboter den Würfel berührt. Jetzt bei
274, und die beiden Kopfkameras sind sich auf 4 Frames einig. Die Onset-Konflikte insgesamt fallen
von 67 auf 38, und Gelb liefert erstmals einen Anker.

### 7.2 Korrektur 2 hat nichts bewirkt — die Zeitpunkt-These war falsch

Die Fingerkuppen wurden auf das Ende der Schließbewegung vorgezogen, im Mittel neun Frames früher.
Die Höhe änderte sich um Millimeter:

| Episode | z bei Onset (Lauf 01) | z bei Schließende (Lauf 02) |
|---:|---:|---:|
| 0 rot | 0,975 | 0,974 |
| 1 rot | 0,985 | 0,988 |
| 7 rot | 0,982 | 0,979 |
| 11 rot | 1,009 | 1,005 |
| 13 rot | 1,039 | 1,043 |
| 15 rot | 1,000 | 0,998 |

Der Höhenversatz stammt also **nicht** aus dem Messzeitpunkt. Die in §3 aufgestellte These ist
damit widerlegt; die z-Beobachtung selbst (Anker über der Würfeloberseite) bleibt bestehen.

### 7.3 Die eigentliche Ursache: die Hand ist beim „Griff" 10 cm offen

Über alle 60 erkannten Schließintervalle des Laufs:

| Größe | Wert |
|---|---:|
| Kuppenöffnung vor dem Intervall | Median 0,112 m |
| Kuppenöffnung nach dem Intervall | Median 0,108 m |
| Intervalle mit Endöffnung ≤ 0,07 m | **1 von 60** |
| Endöffnung der drei akzeptierten Anker | 0,096 / 0,099 / 0,100 m |

Eine DEX3-Hand, die einen 5-cm-Würfel hält, kann keine mittlere Kuppenöffnung von 10 cm haben.
Was `closure_candidates` findet, ist eine Fingerbewegung von 0,6 bis 36 mm an einer **weit offenen**
Hand — kein Zugriff. Der Fingerkuppen-Schwerpunkt ist folglich die Position einer schwebenden
offenen Hand, und die liegt naturgemäß über und vor dem Würfel. Das erklärt beide Versätze aus §3
ohne jede Annahme über das Kameramodell.

Unabhängige Bestätigung aus einem ganz anderen Lauf: der `scan`-Durchgang vom 2026-08-17 misst
dieselbe Größe (`get_contact_points_w`) über die **ganze** Episode und findet bei **101 von 116
Griffen** eine engste Kuppenöffnung über 6 cm ([co-training.md](../training/co-training.md), §3.2a).
Die Öffnung fällt also auch außerhalb des 25-Frame-Fensters nie auf Greifweite. Damit ist die
Fingeröffnung als Greifdetektor für diese Hand ungeeignet — im alten Pfad wie in v4.

### 7.4 Was daraus folgt

`pick_anchored_homography` steht und fällt mit der Annahme „Fingerkuppen-Schwerpunkt beim Zugriff
≈ Würfelposition". Gemessen liegt der Anker 4–11 cm daneben, und der Fehler streut episodenweise.
Das Verfahren zielt auf 1,5 cm Holdout-Median — diese Genauigkeit ist mit diesem Anker nicht
erreichbar, unabhängig von Homographie, Gates und Ausbeute.

Der Gegenentwurf ist bereits vorhanden und in §3 gegengerechnet: das Pinhole-Modell projiziert
dieselben Würfelpixel ohne jeden Anker auf x ∈ [0,23 … 0,41], beide Kameras auf 1,7 cm einig, alle
120 Messungen im Spawnband. Es braucht genau eine Bestätigung, nämlich die Vorwärtsrichtung gegen
ein gerendertes Bild (`server_rl_run.sh layoutcheck`). Fällt die aus, ist die Würfellage direkt aus
dem Realbild bestimmbar und der gesamte Ankerapparat entbehrlich.

---

## 8. `layoutcheck`: das Kameramodell ist bestätigt (2026-08-22, `runs/20260822/03`)

`server_rl_run.sh cams` liefert ein gerendertes Bild je Kopfkamera **und** die tatsächlichen
Würfel-Weltposen aus der Physik. Damit ist die Vorwärtsrichtung prüfbar, ohne die blockierte
Kalibrierung.

Grundwahrheit dieses Laufs (env-lokal): rot (0,363, −0,109, 0,895), grün (0,322, −0,016, 0,901),
gelb (0,321, 0,141, 0,895). **Nicht** die Nominalwerte der Config — die Würfel werden beim Reset
gestreut, und ein erster Durchgang gegen die Nominalwerte ergab irreführende 4,6 cm.

| Kamera | Ebene 0,915 (Code) | Ebene = echte Ruhelage |
|---|---|---|
| cam_left_high | Bias (−1,01, −0,05), Streuung (0,41, 0,39), Median 1,11 cm | Bias (+0,20, −0,12), Streuung (0,46, 0,10), Median **0,30 cm** |
| cam_right_high | Bias (−0,70, −0,02), Streuung (0,70, 0,21), Median 0,89 cm | Bias (+0,51, +0,12), Streuung (0,74, 0,29), Median **0,22 cm** |

Zwei Schlüsse:

1. **Das Pinhole-Modell liefert die Würfellage auf rund 1 cm — mit dem Code, wie er ist.** Das ist
   besser als die 1,5 cm, die v4 selbst anstrebt, und rund zehnmal genauer als die
   Fingerkuppen-Anker (§7.4: 4–11 cm). Der Anker wird nicht gebraucht.
2. **Der verbleibende x-Bias ist eine falsche Ebenenhöhe, kein Modellfehler.** Die Würfel ruhen bei
   z = 0,895; `Z_CUBE_CENTER` / `block_z_surface` rechnen mit 0,915. Ursache: die Tischplatte liegt
   bei 0,870 (Tischmittelpunkt z = 0,435), nicht bei den 0,890, die der Kommentar in
   `g1_dex3_blockstack_env.py:422` annimmt. Die Würfel fallen beim Reset zwei Zentimeter und landen
   auf 0,870 + 0,025 = 0,895. Mit korrigierter Ebene sinkt der Fehler auf 0,2–0,3 cm.

Kein Kameraparameter erklärt den Rest: eine Suche über Brennweite (45–100°), Blickziel und
Kamerahöhe kommt nicht unter ~2 cm, solange die Grundwahrheit falsch ist, und braucht keine
Korrektur, sobald sie stimmt. Auch der Top-Face-Schwerpunkt ist nicht besser als der
Silhouetten-Schwerpunkt (0,99 gegen 1,11 cm) — die aufwendigere Messung aus v4 bringt hier nichts.

### Was damit noch nicht bewiesen ist

Geprüft ist die Konsistenz **innerhalb der Simulation**: Config → Rendering → Rückprojektion. Dass
die **reale** Kopfkamera derselben Pose folgt, bleibt offen; die Pose wurde aus Datensatzbildern
rekonstruiert, nicht kalibriert. Der bisher einzige Beleg aus Realbildern: über 120 Messungen
stimmen beide Kameras auf 1,7 cm überein, und alle Positionen liegen im Spawnband (§3).

### Empfehlung

`pick_anchored_homography` aufgeben. Die Würfellage kommt aus dem Realbild
(`extract_block_layout extract` → `layout.json`, der bestehende Pfad A), mit korrigierter
Ebenenhöhe und dem per `--bias` bestimmten Restversatz. Die brauchbaren Teile von v4 — Gates,
Hash-Kette, Holdout-Logik, Projektionsprüfung im Renderer — bleiben und bekommen die Posen aus
dieser Quelle.
