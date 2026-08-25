# Co-Training auf echten und gerenderten Bildern (Schritt 4)

> **TL;DR:** Schritt 4 der Domain-Gap-Behebung: warum Co-Training auf echten + gerenderten
> Sim-Bildern der nächste Hebel ist, und welche Werkzeuge dafür gebaut wurden (Renderer,
> Zwei-Datensatz-Training `USE_COTRAIN=1`). Werkzeuge stehen, der Trainingslauf steht noch aus.

**Stand:** 2026-08-17 · erster voller Renderlauf gefahren (60 Episoden), **Befund: Frames ab
dem Griff sind falsch beschriftet** → `RENDER_STOP_AT_GRASP` (§ 3.2a), Trainingslauf steht
noch aus · Vorgeschichte:
[Lauf 34](../ergebnisse/diagnose-chronik.md#läufe-3334-runs2026081403-runs2026081404-der-tune_visual-checkpoint-im-closed-loop),
[next-steps.md Schritt 4](../../next-steps.md)

---

## 1. Warum

Die Beweiskette ist geschlossen und sie zeigt in genau eine Richtung:

| Befund | Zahl | Beleg |
|---|---|---|
| Auf **echten** Datensatz-Bildern kommandiert die Politik den vollen Griff | 100 % der Demo | Lauf 33, `span` |
| In der **Sim** kommandiert dieselbe Politik einen Bruchteil davon | 20,5 % → **27,6 %** mit `TUNE_VISUAL` | Läufe 31 / 34 |
| Der Sprung ist echt, nicht Rauschen | vollständige Trennung, p = 3,3 · 10⁻⁴ | Lauf 34, n = 10 gegen 5 |
| Er reicht trotzdem nicht | `lifted` 0/10, beste Anhebung 1,68 cm von 2 cm | Lauf 34 |

Der aufgetaute Vision-Encoder ist also **der wirksame Hebel** — und er ist mit Realbildern
allein ausgereizt. Er hat nie ein Sim-Bild gesehen. Genau das fordert
[`lauf2-vision-auswertung.md` §6.2](../ergebnisse/lauf2-vision-auswertung.md): `tune_visual`
lohnt nur, wenn der Encoder sim-ähnliche Bilder im Training bekommt.

Der Weg dahin führt nicht über mehr Realdaten, sondern über **dieselben Aktionen mit
Sim-Bildern**: Wir spielen aufgezeichnete Episoden in Isaac Lab ab und zeichnen dabei die
vier Policy-Kameras auf. Ergebnis sind Paare *(Sim-Bild, echte Aktion)* — für jede
gerenderte Episode existiert die reale Zwillings-Episode mit identischen Aktionen. Ein
Encoder, der auf beiden dieselbe Aktion produzieren muss, kann sich nicht mehr auf die
Oberflächen-Statistik der Realbilder verlassen.

---

## 2. Was dafür gebaut wurde

| Datei | Rolle |
|---|---|
| [`Simulation/g1_dex3_sim/render_cotrain_dataset.py`](../../Simulation/g1_dex3_sim/render_cotrain_dataset.py) | Der Renderer. Zwei Stufen (`scan` → `render`), schreibt einen vollwertigen LeRobot-**v2.1**-Datensatz. |
| [`Simulation/server_rl_run.sh render`](../../Simulation/server_rl_run.sh) | Wrapper auf dem Sim-Server: Checkpoint + Asset + echter Datensatz sicherstellen, beide Stufen fahren. |
| [`Training/scripts/launch_cotrain.py`](../../Training/scripts/launch_cotrain.py) | Trainings-Einstieg für **zwei** Datensätze mit Mischungsverhältnis. |
| [`Training/scripts/run_finetuning_cotrain.sh`](../../Training/scripts/run_finetuning_cotrain.sh) | Trainings-Launcher (eigener Namespace `blockstacking_cotrain`, `--tune_visual`, Split an). |
| [`Training/scripts/entrypoint.sh`](../../Training/scripts/entrypoint.sh) | `USE_COTRAIN=1` routet auf den neuen Launcher. |

**Kein Submodul-Eingriff.** Die Misch-Fähigkeit steckt bereits im Fork
(`SingleDatasetConfig.mix_ratio`, `ShardedMixtureDataset`); nur der CLI-Einstieg
`gr00t/experiment/launch_finetune.py` verdrahtet genau einen Datensatz mit `mix_ratio: 1.0`.
`launch_cotrain.py` ist dessen Kopie mit **einer** Abweichung — der Datensatz-Liste. Damit
entfallen Fork-Push, Dockerfile-Pin und Image-Rebuild; `Training/scripts/` wird auf KISSKI
ohnehin als `/scripts` eingehängt.

### Warum der Renderer zwei Stufen hat

Der Datensatz speichert **keine Objektposen**. Spielt man die Aktionen ab, während die Env
ihre Würfel zufällig auslegt, entstehen Paare, in denen der Arm dorthin greift, wo kein
Würfel liegt — und der Encoder lernt, den Würfel zu **ignorieren**. Das wäre schlimmer als
gar nichts. Die Würfel müssen also dorthin, wo sie in der echten Aufnahme lagen.

Die beiden Stufen heute:

1. **`scan`** — Episode abspielen, Kameras auf ein Zehntel der Auflösung. Liefert je Episode
   den Arm-Tracking-Fehler und je Hand einen Greifpunkt aus der Fingerkinematik → `scan.json`.
   Der Greifpunkt platziert seit 2026-08-22 **keine Würfel mehr**: er liegt bei ~der Hälfte
   der Griffe auf dem Transportweg statt am Pick (§ 3.2a), und die Fingeröffnung ist für
   diese Hand überhaupt kein Greifdetektor (bei 101 von 116 Griffen bleibt die engste
   Öffnung über 6 cm, bei 5 cm Würfelkante). Aus `scan.json` kommt nur noch der
   Arm-Tracking-Fehler. Die Würfellage kommt ausschließlich aus den **Realbildern** — § 3.0
   (`layout`,
   [wuerfellage-rekonstruktion.md](../simulation/wuerfellage-rekonstruktion.md)). Die
   ursprüngliche Greifpunkt-Begründung ist in [../historie.md](../historie.md) dokumentiert.
2. **`render`** — dieselben Episoden mit den Würfeln an den Layout-Positionen (x/y aus dem
   Realbild, z = Tischauflage) und in kalibrierter Auflösung 640 × 480. → der Datensatz

> **Korrektur nach dem ersten Rauchtest (2026-08-14):** Der Scan ist billiger als das
> Rendern, aber **nicht** vernachlässigbar. Gemessen: ~12 Steps/s bei 64×64 gegen ~8,6
> Steps/s im Eval bei voller Auflösung. Die Physik (7 Sub-Steps à 5 ms je Policy-Step)
> dominiert, nicht das Rendering. Die 94-%-Zahl aus
> [live-ansicht.md](../simulation/live-ansicht.md) meint „Sim **und** Render" zusammen und
> sagt über die Aufteilung darin nichts — ich hatte sie hier zunächst falsch gelesen.
> Realistisch sind **~3 min je Episode für beide Stufen**.

---

## 3. Ablauf

### 3.0 Zuerst die Würfellage aus den Realbildern — `layout` (seit 2026-08-17)

**Das ist der wichtigste Schritt, und er war zuerst nicht da.** Seit 2026-08-22 ist er die
einzige Quelle: fehlt für eine Episode die Lage auch nur eines Würfels, verwirft `render` die
Episode (`status: rejected_layout`), statt sie mit einem Greifpunkt oder einer Zufallslage zu
füllen. Beide Rückfallebenen erzeugten Bilder, auf denen der Arm an einem Würfel vorbeigreift,
der dort nie lag — und das sieht aus wie gültige Aufsicht.

Der Verzicht kostet praktisch nichts: der Layout-Lauf vom 2026-08-22 findet in **60 von 60
Episoden alle drei Würfel**, die beiden Kopfkameras sind sich im Median auf 1,17 cm einig
(max 2,31 cm), und das Kameramodell selbst trifft auf 0,3 cm
([Läufe 35–37](../ergebnisse/diagnose-chronik.md)).

```bash
RENDER_EPISODES=60 LAYOUT_OVERWRITE=1 ./Simulation/server_rl_run.sh layout
```

> **Verfahren, Koordinatentransformation und offene Punkte im Detail:**
> [`../simulation/wuerfellage-rekonstruktion.md`](../simulation/wuerfellage-rekonstruktion.md)
> — das Übergabedokument dazu.

[`extract_block_layout.py`](../../Simulation/g1_dex3_sim/extract_block_layout.py) liest den
ersten Frame jeder Episode, segmentiert die drei gesättigten Würfel (rot = `block_0`,
grün = `block_1`, gelb = `block_2`) gegen den weißen Tisch und schneidet den Strahl durch
jeden Blob-Schwerpunkt mit der Würfelebene z = 0,915. Ergebnis: `layout.json` mit x/y
**aller drei** Würfel je Episode — aus dem Bild gelesen, nicht aus der Fingerbewegung
rekonstruiert. Kein Isaac, keine GPU, Minuten statt Stunden.

**Vorher das Kameramodell gegen den Renderer prüfen.** Eine Rückprojektion ist nur so gut
wie die Annahme, dass Isaac mit der konfigurierten Pose auch rendert — in Lauf 13
(2026-08-08) lagen USD-Stage und `cam.data` 95,6° auseinander und drei Sim-Läufe waren
umsonst. Deshalb:

```bash
# cubes_xyz einer schon gerenderten Episode aus render_manifest.json holen, Frame 0 daneben
LAYOUTCHECK_FRAME=/data/cotrain/g1_dex3_rendered/videos/chunk-000/observation.images.cam_left_high/episode_000000.mp4 \
LAYOUTCHECK_EXPECT='[[…],[…],[…]]' ./Simulation/server_rl_run.sh layoutcheck
```

Lesart: **Mittel** = Bias des Schätzers (der Blob-Schwerpunkt ist der Schwerpunkt der
sichtbaren Flächen, nicht die Projektion des Würfelmittelpunkts) → per `LAYOUT_BIAS="dx dy"`
in Metern herausrechnen. **Streuung** = der Rest, und erst die entscheidet, ob das Layout
brauchbar ist. Über ~2 cm Streuung ist es das nicht.

Lokal vorab geprüft (2026-08-17, gegen [`camera_reference/`](../../Simulation/camera_reference/)):

| Probe | Ergebnis |
|---|---|
| Rückprojektion ↔ Projektion | exakter Roundtrip auf der Würfelebene |
| Bildaufteilung | Tisch hinten 1 %, vorn 97 % der Bildhöhe (dokumentiert: 5 % / 99 %) |
| Skala am bekannten 5-cm-Würfel | **5,2 cm quer** im Mittel über 6 Messungen |
| Detektion | 3/3 Würfel in beiden High-Kameras, Kreuze auf den Würfeln |
| Beide Kameras einig bis | 1,3–2,3 cm |

Die Skalenprobe ist die belastbare: eine bekannte Länge geht durch das Modell und kommt
richtig heraus. Was sie **nicht** ausschließt, ist ein globaler Versatz — den findet nur
`layoutcheck` gegen ein gerendertes Bild.

> **Nebenbefund:** das schwarze Stapelband ist in der Sim 12 cm breit bei (0,35 / 0,00)
> konfiguriert; aus den Realbildern zurückgerechnet sind es **7,2 cm bei (0,31 / +0,06)**.
> Die Sim-Werte sind im Code als Näherung markiert — hier steht jetzt eine Messung dagegen.

### 3.1 Rauchtest zuerst (≈ 2 min) — prüft die Klempnerei, nicht die Inhalte

Zwei Episoden à 60 Frames. Prüft Isaac-Start, Würfelsetzen, MP4-Schreiber, Parquet und
`meta/` — ohne Stunden zu investieren.

```bash
HF_TOKEN=hf_... RENDER_EPISODES=2 RENDER_MAX_FRAMES=60 \
    ./Simulation/server_rl_run.sh render
```

Erwartete Ausgabe am Ende beider Stufen: `[scan] fertig.` / `[render] fertig.` und
`meta/ geschrieben: 2 Episoden, 120 Frames`.

> **Was dieser Test NICHT zeigt: ob die Greifpunkte stimmen.** 60 Frames sind 2 Sekunden
> einer 30-sekündigen Episode — dort hat noch keine Hand gegriffen. Meldungen wie
> „kein Greifpunkt — Hand schließt nicht (0.1 cm)" sind hier normal und **kein** Befund.
>
> Die 60-Frame-Stummel bleiben liegen; der Fortsetz-Mechanismus vergleicht seit
> 2026-08-14 aber die gespeicherte Länge mit der Quell-Länge und rendert sie beim
> echten Lauf von selbst neu („vorhandene Fassung hat 60 statt 934 Frames … wird neu
> gerendert"). Wer sauber anfangen will: `rm -rf <data>/cotrain/g1_dex3_rendered`.

### 3.1b Dann der Scan über den ganzen Satz — das ist die inhaltliche Prüfung

Der Scan rendert nichts Brauchbares und ist der billigere Teil, liefert aber die
entscheidende Zahl: bei wie vielen Episoden wird überhaupt ein Greifpunkt gefunden?

```bash
HF_TOKEN=hf_... RENDER_EPISODES=60 RENDER_STAGE=scan \
    ./Simulation/server_rl_run.sh render
```

Danach `scan.json` durchsehen, **bevor** die Renderstunden laufen:

```bash
python3 - <<'EOF'
import json
s = json.load(open('<data>/cotrain/g1_dex3_rendered/scan.json'))['episodes']
ok = [e for e, v in s.items() if any(h.get('ok') for h in v['hands'])]
print(f"{len(ok)}/{len(s)} Episoden mit Greifpunkt")
print("Tracking max:", max(v['arm_tracking_error_rad'] for v in s.values()))
for e, v in list(s.items())[:5]:
    print(e, [h.get('xy') for h in v['hands'] if h.get('ok')])
EOF
```

Plausibel sind x um 0,30–0,50 und y um ±0,20. Liegt die Trefferquote deutlich unter
etwa der Hälfte, erst die Erkennung nachschärfen (`--min-close`-Logik in
`find_grasp_points`), statt Material mit zufällig liegenden Würfeln zu erzeugen.

> **`ok` in `scan.json` ist ein Greifpunkt-Detektor, kein Erfolgsdetektor.** Es heißt „die
> Hand hat sich um ≥ 2 cm geschlossen, und hier waren die Kuppen dabei" — nichts darüber, ob
> gestapelt wurde. Der Erfolg der Aufnahme ist eine Eigenschaft des **realen** Datensatzes
> und im Replay nicht messbar. Wer `ok: true` als „gute Demo" liest, kommt zwangsläufig zu
> dem Schluss, der QA-Scan lüge. Die Zahl, die den gerenderten Satz beurteilt, steht in
> `render_manifest.json` → `consistency` (§ 3.2b).

### 3.2 Der lange Lauf

```bash
HF_TOKEN=hf_... RENDER_EPISODES=60 ./Simulation/server_rl_run.sh render
```

**Fortsetzbar.** Fertige Episoden werden übersprungen; ein Abbruch kostet höchstens die
angefangene. Mehr Episoden nachschieben geht jederzeit mit einem höheren
`RENDER_EPISODES` — der Lauf rendert dann nur die neu hinzugekommenen. In `tmux`/`screen`
legen, sonst beendet ein Verbindungsabbruch den Lauf.

### 3.2a Nur bis zum Griff rendern — `RENDER_STOP_AT_GRASP` (Default an, seit 2026-08-17)

**Der Befund aus dem ersten vollen Lauf (60 Episoden, 61 203 Frames).** Sichtprüfung der
Endframes zeigte: die Würfel liegen am Episodenende unverändert verstreut, kein Stapel. Das
ist kein Datenfehler, sondern die Vorhersage des Aufbaus — aus zwei Gründen:

| Grenze | Beleg |
|---|---|
| **Der Greifpunkt ist nicht der Pick.** `np.argmin` sucht das Minimum der Fingeröffnung über die **ganze** Episode. Beim Pick-and-Place bleibt die Hand vom Zugreifen bis zum Ablegen geschlossen — ein breites Tal, kein Ausschlag; wo darin das Minimum liegt, entscheidet minimales Nachdrücken. **48 von 116 Griffen liegen jenseits von 60 % der Episode, 21 jenseits von 80 %.** Der Würfel landete damit auf dem Transportweg, und der Arm griff beim echten Pick ins Leere — in x, y **und** z, weil `place_cubes` nur x/y nimmt und z auf die Tischauflage setzt. **Behoben durch § 3.0** (Lage aus dem Realbild). | `scan.json` des Laufs 2026-08-17, `close_step / length` |
| `find_grasp_points` liefert per `np.argmin` **genau einen** Greifpunkt je Hand. „Stack three block" braucht zwei bis vier Pick-and-Place-Zyklen. Für jeden Griff außer einem pro Hand liegt also kein Würfel. Würfel 2 wurde ohnehin zufällig „daneben" gelegt. | [`render_cotrain_dataset.py`](../../Simulation/g1_dex3_sim/render_cotrain_dataset.py) `find_grasp_points`, `place_cubes` |
| Der Würfel wird **einmal** gesetzt, danach entscheidet die Kontaktphysik — und die greift im Replay meist nicht: bei **101 von 116 Griffen** liegt die engste erreichte Kuppenöffnung über **6 cm**, bei 5 cm Würfelkante. Die Hand schließt sich *neben* dem Würfel. | `scan.json` des Laufs 2026-08-17, `spread_min_cm` |

Damit sind Frames **vor** dem Griff brauchbar (der Würfel liegt dort, wo der Arm hinfährt)
und Frames **ab** dem Griff **falsch beschriftet** (Bild: Würfel auf dem Tisch, Aktion:
Würfel transportieren). Falsch beschriftete Paare sind schlimmer als fehlende.

`RENDER_STOP_AT_GRASP=1` (Default) schneidet jede Episode an der **ersten Würfelbewegung**
ab — nicht am spätesten Würfel: ein bewegter Würfel ist in beiden Kopfkameras zu sehen, also
verdirbt er auch die Frames der anderen Hand.

> **Seit 2026-08-22 kommt diese Grenze aus `layout.json` (`motion_onset`), nicht mehr aus
> `close_step`.** Der alte Weg war zweimal falsch. Erstens greift der Detektor für diese Hand
> nicht: bei 101 von 116 Griffen bleibt die engste Kuppenöffnung über 6 cm bei 5 cm
> Würfelkante. Zweitens lag er, wo er etwas fand, zu spät — in Episode 0 endete das Fenster
> bei Frame 136, während sich der rote Würfel real ab Frame 108 bewegt: **28 Frames, 21 % der
> Episode, falsch beschriftet**. Der Bewegungsbeginn misst direkt, was das Fenster braucht,
> und wird nur gezählt, wenn beide Kopfkameras sich auf 12 Frames einig sind. Episoden ohne
> einen einzigen belastbaren Onset werden verworfen (`skipped_window`).

`RENDER_GRASP_WINDOW=N` rendert nur die letzten N Frames davor. Das schneidet den
Leerlauf-Kopf langer Aufnahmen weg und vereinheitlicht das Gewicht der Episoden — ohne das
stellt Episode 170 mit ihren 3433 Vorlauf-Frames allein 14 % des Satzes.

| `RENDER_GRASP_WINDOW` | Episoden | Frames | Renderzeit (8,6 Steps/s) |
|---|---|---|---|
| `0` (ab Frame 0) | 58 | 24 352 | ~50 min |
| `600` ★ empfohlen | 58 | 19 579 | ~40 min |
| `300` | 58 | 12 871 | ~28 min |

Zwei Episoden fallen durch `RENDER_MIN_WINDOW=60` heraus (126 und 150, Griff bei Frame 56
bzw. 25) — dort startet die Hand geschlossen, das Minimum der Öffnungsspur ist kein
Greifmoment.

**Das ist die Zwischenlösung, nicht das Ziel.** Richtig wird es mit Greif-*Intervallen*
statt -Punkten plus kinematischem Attach: die Würfelpose zwischen `close` und `release`
jeden Step auf den Kuppen-Schwerpunkt schreiben, bei `release` fallen lassen. Dann sind
auch Transport- und Stapelphasen verwendbar und der Satz wächst von 40 % auf ~100 % der
Frames. Noch nicht gebaut.

### 3.2b Die QA-Zahl: Bild-Aktions-Konsistenz, nicht Aufgabenerfolg

`render_manifest.json` enthält seit 2026-08-17 je Episode ein `consistency`-Feld, und der
Lauf fasst es am Ende zusammen. Gemessen wird je Hand der Abstand
**Kuppen-Schwerpunkt ↔ nächster Würfelmittelpunkt im Moment des engsten Griffs**, dazu je
Würfel Bewegung und Anhebung.

Das ist die richtige Frage: der Würfel wurde per Konstruktion unter den Greifpunkt gelegt,
dort gehört ein kleiner Wert hin. Ein großer Wert heißt, die Hand schließt sich neben dem
Würfel — ab diesem Frame beschreibt die Aktion einen Transport, den das Bild nicht zeigt.

```bash
python3 - <<'EOF'
import json, statistics as st
m = json.load(open('<data>/cotrain/g1_dex3_grasp/render_manifest.json'))['episodes']
d = [h['dist_cm'] for r in m.values() if r.get('consistency')
     for h in r['consistency']['hands'] if h]
print(f"{len(d)} Griffe, Median {st.median(d):.1f} cm, <=4 cm: {sum(x<=4 for x in d)}")
EOF
```

Erwartung bei korrekter Platzierung: Median deutlich unter 4 cm (Würfel-Halbdiagonale
4,3 cm). Liegt er darüber, stimmt die Greifpunkt-Rekonstruktion nicht und Rendern hilft
nicht — dann erst `find_grasp_points` reparieren.

### 3.3 Zum Trainings-Rechner bringen

Gerendert wird auf der RT-Core-GPU, trainiert auf KISSKI — das Material muss also
transportiert werden. Größenordnung: **1–2 GB für 60 Episoden** (4 Kameras, h264).

```bash
huggingface-cli upload <user>/g1-dex3-rendered <data>/cotrain/g1_dex3_rendered \
    --repo-type dataset --private
```

### 3.4 Trainieren

```bash
export HF_TOKEN=hf_... WANDB_API_KEY=...
export USE_COTRAIN=1
export COTRAIN_HF_REPO=<user>/g1-dex3-rendered   # oder COTRAIN_DATASET_PATH lokal
export COTRAIN_MIX_RATIO=0.25
export MAX_STEPS=30000        # Lauf 3: die letzten 14 000 Steps haben verschlechtert
export GLOBAL_BATCH_SIZE=32
sbatch --export=ALL Training/kisski_submit.sh
```

> **Kein Image-Rebuild nötig, auf beiden Seiten.** `Training/scripts/` ist auf KISSKI als
> `/scripts` eingehängt, `Simulation/g1_dex3_sim/` auf dem Sim-Server (read-only) als
> `/workspace/g1_dex3_sim` — ein `git pull` auf dem jeweiligen Rechner genügt.
>
> **`kisski_submit.sh` reicht Env-Vars über eine explizite Liste durch**, nicht pauschal.
> Die vier `COTRAIN`-Variablen stehen seit 2026-08-14 darin; ohne diesen Eintrag wäre
> `USE_COTRAIN` still verschluckt worden und der Job hätte ein normales Vision-Training
> gefahren. Die Startausgabe zeigt `USE_COTRAIN: 1  (Mischung 0.25 gerendert)` — **prüfen**.
> Die Lernrate fällt dabei automatisch auf den Vision-Wert (1e-4, Warmup 0,1), weil das
> Co-Training den Encoder auftaut.

### 3.5 Auswerten — zwei Messungen, nicht eine

```bash
# a) Realdomäne: hat das gerenderte Material der echten Aufgabe geschadet?
export RUN_DIR=/data/g1_dex3_finetune/blockstacking_cotrain
sbatch Training/kisski_open_loop_eval.sh

# b) Simdomäne: greift die Politik jetzt mehr? (identisches Protokoll wie Lauf 34!)
NUM_EPISODES=10 EPISODE_LENGTH_S=40 CHECKPOINT_PATH=<bester ckpt> \
    HF_TOKEN=hf_... ./Simulation/server_rl_run.sh eval
```

`EPISODE_LENGTH_S=40` ist Pflicht und nicht verhandelbar: die Fingerspanne ist ein Maximum
über die Episode, ein längeres Fenster hebt sie allein dadurch. Genau daran ist Lauf 33 als
Vergleich gescheitert.

---

## 4. Die beiden Entscheidungen — und wie sie begründet sind

`next-steps.md` hatte sie als offene Vorfragen notiert. Hier die Antworten.

### 4.1 Wie viele Episoden? → **60 für den ersten Lauf**

| | |
|---|---|
| Dauer je Episode | ~3 min für **beide** Stufen (Scan ~12 Steps/s, Render ~9 Steps/s, mittlere Episode 934 Frames) |
| 60 Episoden | ≈ 3 h Wanduhr, ≈ 56 000 Frames |
| Alle 240 Trainings-Episoden | ≈ 12 h, ≈ 224 000 Frames |

60 ist kein Optimum, sondern der Punkt, an dem sich der erste Lauf **innerhalb eines
Abends** erzeugen lässt. Weil der Renderer fortsetzbar ist, ist die Zahl keine
Einbahnstraße: Bringt der Lauf etwas, aber zu wenig, wird auf 120 aufgestockt, ohne die
ersten 60 neu zu rendern.

Die Episoden werden **gleichmäßig über den Trainingsbereich gestreut** (0, 4, 8, …), nicht
als Präfix genommen — die Aufnahmen sind chronologisch, ein Präfix wäre eine Tageszeit.

**Test-Episoden werden nie gerendert.** Der Renderer zieht die Grenze mit derselben Formel
wie [`lib_split.sh`](../../Training/scripts/lib_split.sh) (`int(total × ratio)` = 240) und
bricht ab, wenn `--episode-ids` eine zurückgehaltene Episode enthält. Andernfalls sähe das
Modell die Testepisoden in gerenderter Form, und die Validierungs-MSE des
Checkpoint-Sweeps — die einzige Zahl, die Overfitting sichtbar macht — wäre wertlos.

### 4.2 Welches Mischungsverhältnis? → **0,25 gerendert**

Zwei unabhängige Überlegungen laufen auf denselben Bereich zu:

**(a) Gleiche Wiederholungsrate.** `mix_ratio` ist eine Sampling-Wahrscheinlichkeit, kein
Längenverhältnis. Bei 60 gerenderten (56 k Frames) gegen 240 echte (224 k Frames) sieht das
Modell jedes gerenderte Bild um den Faktor 4 häufiger als jedes echte, wenn beide gleich
gewichtet werden — der kleinere Satz wird auswendig gelernt. Gleich häufig ist jedes Bild
bei

```
mix* = F_gerendert / (F_gerendert + F_echt) = 56 000 / 280 000 ≈ 0,20
```

**(b) Verhaltensbreite.** Die gerenderten Episoden tragen **keine neuen Aktionen** — sie
sind die Zwillinge von 60 der 240 Trainingsepisoden. Bei `mix = 0,5` verbrächte das Modell
die Hälfte aller Updates auf einem Viertel des Bewegungsrepertoires. Das ist genau die
Verengung, die Lauf 2 gezeigt hat (Kollaps auf reines Arm-Zurückziehen), nur mit
umgekehrtem Vorzeichen.

**0,25** liegt knapp über dem Ausgleichspunkt — ein bewusster kleiner Schubs Richtung Sim,
ohne die Realdomäne zu verdrängen. Wer mehr Sim-Anteil will, erzeugt **mehr Episoden**
statt das Verhältnis zu erhöhen; die Formel oben liefert dann den passenden Wert.

---

## 5. Vorregistrierte Regel — was den Lauf zum Erfolg oder zum Ende macht

Vor dem Lauf festgeschrieben, damit hinterher nicht die Interpretation wandert.

| Messung | Referenz | Deutung |
|---|---|---|
| **Fingerspanne Sim** (10 Ep. × 40 s) | Lauf 34: Median 0,577 rad = 27,6 % | **Primäres Gate.** Deutlich darüber → der Weg trägt, aufstocken. Auf dem Niveau → Sim-Bilder allein genügen nicht, RL rückt vor. |
| **Validierungs-MSE echt** (Checkpoint-Sweep) | Lauf 3: 0,00716 | **Leitplanke.** Merklich schlechter → das gerenderte Material verdrängt echtes Lernen, `COTRAIN_MIX_RATIO` senken statt weiterzurendern. |
| **`lifted`** | Lauf 34: 0/10, best 1,68 cm | Erster echter Verhaltensfortschritt. Die Schwelle liegt nur 0,32 cm über dem heutigen Bestwert. |
| **Domain-Gap** (`server_rl_run.sh gap`) | Mittel 0,22, `cam_left_wrist` 0,36 | Kontrolle, ob sich die Encoder-Repräsentation wirklich angenähert hat. |

Die Kopplung Fingerspanne ↔ Anhebung war in Lauf 34 **Spearman +0,62** — die Spanne
reagiert also früher als die Erfolgsrate und ist damit das empfindlichere Gate.

---

## 6. Was schiefgehen kann

Ehrlich benannt, weil jeder dieser Punkte den Lauf entwerten würde:

1. **Die Greifpunkt-Heuristik ist eine Rekonstruktion, keine Messung.** Der Datensatz
   enthält keine Objektposen; wo der Würfel lag, wird aus der Handbewegung erschlossen.
   Schließt eine Hand nie sichtbar (< 2 cm Änderung der Fingeröffnung), bekommt sie keinen
   Würfel und die Episode ist visuell schwächer gekoppelt. `scan.json` protokolliert das je
   Hand — **vor** dem langen Lauf durchsehen.
2. **Nur x/y kommen aus dem Greifpunkt, z ist die Tischauflage.** Richtig für Würfel, die
   auf dem Tisch liegen; falsch für den zweiten Teil eines Stapelvorgangs, bei dem der
   Zielwürfel erhöht steht. Diese Phase ist im gerenderten Material nicht korrekt abgebildet.
3. **Tracking-Filter.** Episoden, deren Arme den Aktionen im Mittel schlechter als
   0,15 rad folgen, werden verworfen (`--tracking-error-max`) — ihr Bild zeigt eine andere
   Pose als die Aktion beschreibt. Verwirft der Lauf viele Episoden, ist nicht der Filter
   das Problem, sondern die Physik-Konfiguration.
4. **Der dritte Würfel liegt zufällig.** Bei zwei erkannten Greifpunkten bleibt Würfel 2
   ein Ablenker im konfigurierten Band, mit Mindestabstand zu den gesetzten.
5. **Domain Randomization ist beim Rendern an** (`DR_ENABLED=1`, Default). Das ist
   erwünscht — es verbreitert die Beleuchtung im Trainingsmaterial. Für einen
   Diagnose-Lauf mit fester Optik `DR_ENABLED=0` setzen.

---

## 7. Env-Vars

### Rendern (Sim-Server)

| Variable | Default | Bedeutung |
|---|---|---|
| `RENDER_EPISODES` | `60` | Anzahl Episoden, gleichmäßig über den Trainingsbereich |
| `RENDER_OUT` | `/data/cotrain/g1_dex3_rendered` | Zielverzeichnis |
| `RENDER_STAGE` | `both` | `scan`, `render` oder beides |
| `RENDER_MAX_FRAMES` | `0` | `>0` kürzt jede Episode (Rauchtest) |
| `RENDER_EPISODE_IDS` | — | Explizite Indices statt Streuung, z. B. `"0 4 8"` |
| `RENDER_OVERWRITE` | `0` | `1` = fertige Episoden neu rendern |
| `DR_ENABLED` | `1` | Beleuchtungs-/Farb-Randomisierung beim Rendern |
| `SPAN_DATASET` | `/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset` | Quelldatensatz (wird bei Bedarf geholt + konvertiert) |

### Trainieren

| Variable | Default | Bedeutung |
|---|---|---|
| `USE_COTRAIN` | `0` | `1` = Co-Training statt Standard-/Vision-Lauf |
| `COTRAIN_DATASET_PATH` | `/data/cotrain/g1_dex3_rendered` | Gerenderter Datensatz im Container |
| `COTRAIN_HF_REPO` | — | HF-Dataset-Repo; wird geladen, wenn der Pfad leer ist |
| `COTRAIN_MIX_RATIO` | `0.5` im Skript, **0.25 empfohlen** (§4.2) | Anteil gerenderter Stichproben |
| `TRAIN_TEST_SPLIT` | `1` | Hier standardmäßig **an** — ohne Split fehlt die Leitplanke aus §5 |

> Der Skript-Default für `COTRAIN_MIX_RATIO` steht auf 0.5, weil das der neutrale Wert für
> „zwei gleichberechtigte Datensätze" ist. Für den konkreten ersten Lauf ist 0.25 die
> begründete Wahl — deshalb im Aufruf explizit setzen.

---

## 8. Verwandte Dokumente

- [next-steps.md](../../next-steps.md) — die Priorisierung, aus der dieser Schritt kommt
- [diagnose-chronik.md, Läufe 33/34](../ergebnisse/diagnose-chronik.md#läufe-3334-runs2026081403-runs2026081404-der-tune_visual-checkpoint-im-closed-loop) — die Messung, die ihn begründet
- [lauf2-vision-auswertung.md](../ergebnisse/lauf2-vision-auswertung.md) — warum `tune_visual` auf reinen Realdaten kollabiert
- [lauf3-vision-split-auswertung.md](../ergebnisse/lauf3-vision-split-auswertung.md) — Checkpoint-Auswahl und Overfitting-Nachweis
- [train-test-split.md](train-test-split.md) — Split-Mechanik und Checkpoint-Sweep
- [env-vars.md](env-vars.md) — vollständige Variablenreferenz
