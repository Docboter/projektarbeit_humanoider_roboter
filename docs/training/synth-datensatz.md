# Fremder Sim-Datensatz im Co-Training (`Cube_Stacking_synth`)

> **TL;DR:** Der extern aufgenommene Sim-Datensatz `Fichtl00/Cube_Stacking_synth` (16 Episoden,
> in der Sim teleoperierter G1) passt schematisch **nicht** zum echten Datensatz — 57-Dim-State
> statt 28, 50 fps statt 30, und die Armachsen tragen eine kartesische EEF-Pose statt
> Gelenkwinkeln. `harmonize_synth_dataset.py` misst die Zuordnung, statt sie zu raten.
> Die Gelenkreihenfolge ist **aufgelöst**: sie steht in den eigenen Sim-Logs, weil der Datensatz
> in dieser Umgebung aufgenommen wurde (§ 5) — aus den Daten allein wäre sie nicht bestimmbar
> gewesen (§ 3.1).
> Nicht hier: das Co-Training-Verfahren selbst ([co-training.md](co-training.md)).

**Stand:** 2026-08-28 · Werkzeug gebaut und abgenommen; Arme, Zeitversatz, Vorzeichen und
Fingerreihenfolge belegt. Der Trainingslauf kann laufen.

---

## 1. Warum dieser Datensatz

Die Beweiskette aus [next-steps.md](../../next-steps.md) endet an einer Stelle: Der aufgetaute
Vision-Encoder ist der wirksame Hebel (Lauf 34: Fingerspanne 20,5 % → 27,6 %, p = 3,3 · 10⁻⁴),
und er ist mit Realbildern allein ausgereizt, weil er **nie ein Sim-Bild gesehen** hat.
[co-training.md](co-training.md) löst das über einen selbst gerenderten Zwillingsdatensatz.

`Fichtl00/Cube_Stacking_synth` ist eine zweite, unabhängige Quelle desselben Materials: ein G1
mit demselben Aufbau, in der Sim teleoperiert, dieselbe Aufgabe (rot unten, gelb Mitte, grün
oben), dieselben vier Policy-Kameras. Er kostet keinen Renderlauf und bringt eine Varianz mit,
die der eigene Renderer nicht hat — andere Bediener, andere Trajektorien, andere Szenen.

**Vorbehalt, der ins Ergebnisprotokoll gehört:** Die Zieldomäne der Messung ist *unsere*
Isaac-Lab-Umgebung. Welcher Simulator und welche Kamerakalibrierung hinter `Cube_Stacking_synth`
stehen, ist nicht dokumentiert. Sim-Bilder allgemein zu sehen macht den Encoder robuster; dass es
*genau* die Lücke zu unserer Sim schließt, ist damit nicht gesagt. Wer das vorab beziffern will,
hält Standbilder des synth-Datensatzes mit
[`measure_domain_gap.py`](../../Simulation/scripts/measure_domain_gap.py) gegen Frames unserer
Sim — dieselbe Kosinus-Metrik wie in der [Domain-Gap-Analyse](../ergebnisse/domain-gap-analyse.md).

---

## 2. Was nicht passt

| Feld | echt (`unitreerobotics/G1_Dex3_BlockStacking_Dataset`) | synth (`Fichtl00/Cube_Stacking_synth`) |
|---|---|---|
| `observation.state` | **28** Dims — `left_arm`, `right_arm`, `left_dex3`, `right_dex3` | **57** Dims — `robot_joint_pos`(43) + `left/right_eef_pos/quat` |
| `action`, Arme | 14 Gelenkwinkel | **je 3 Position + 4 Quaternion** — eine EEF-Pose |
| `action`, Hände | 14 Gelenkwinkel | 14 Gelenkwinkel (Schlüssel `…_hand` statt `…_dex3`) |
| fps | 30 | **50** |
| Annotation | `human.task_description` ← `task_index` | `human.action.task_description` + `human.validity` |
| Format | v3.0, wird zu v2.1 konvertiert | v2.0 mit v2.1-Pfadschablonen |
| Umfang | 301 Ep. / 281 196 Frames / ⌀ 31,1 s | 16 Ep. / 5 325 Frames / **⌀ 6,7 s** |
| Codec | av1 | mp4v |

Vier dieser Zeilen sind Umbenennung oder Umrechnung. Zwei sind echte Arbeit:

**Der Aktionsraum der Arme.** Die Armachsen des synth-Datensatzes heißen `left_arm`/`right_arm`,
enthalten aber keine Gelenkwinkel: die Norm von `action[3:7]` und `action[10:14]` ist über jeden
Frame exakt 1 (größte Abweichung 1,4 · 10⁻⁷). Das ist ein Einheitsquaternion. Dazu passen die
Positionsanteile — z ≈ 0,88…1,15 m (Tischhöhe), links y > 0, rechts y < 0 — und der Abgleich
gegen `left/right_eef_pos` im State (RMSE 2,6…2,9 cm). Teleoperiert wurde also über eine IK auf
Endeffektor-Posen; die Gelenkziele, die der Regler daraus errechnet hat, stehen nicht im
Datensatz. Sie werden ersetzt, siehe § 4.

**Die Bildrate.** `delta_indices` zählt in *Frames*, nicht in Sekunden — die Aktionskette umfasst
16 Frames, bei 30 fps also 533 ms, bei 50 fps nur 320 ms. Ungefixt sähe das Modell dieselbe
Bewegung in zwei verschiedenen Zeitmaßstäben, und die relativen Armdeltas des synth-Anteils wären
systematisch um den Faktor 0,6 zu klein.

Nebenbei: [`run_finetuning_cotrain.sh`](../../Training/scripts/run_finetuning_cotrain.sh) bricht
ohnehin ab, wenn die beiden `meta/modality.json` nicht **byteidentisch** sind (`cmp -s`).

---

## 3. Was gemessen wurde

[`harmonize_synth_dataset.py inspect`](../../Training/scripts/harmonize_synth_dataset.py) rät
nichts. Stand der Messungen vom 2026-08-28:

| Frage | Antwort | Beleg |
|---|---|---|
| Aktionsraum der Arme | kartesische EEF-Pose | Quaternionsnorm exakt 1,0 (max. Abweichung 1,4 · 10⁻⁷) |
| Welche Gelenke sind die Arme | links `[11,15,19,21,23,25,27]`, rechts `[12,16,20,22,24,26,28]` | 15 blockierte Kandidaten = Hüften/Waist/Knie/Sprunggelenke; erschöpfende Suche über alle C(14,7) = 3432 Aufteilungen; FK-R² **0,992/0,992** gegen 0,605/0,596 für den jeweils fremden Armsatz (Kontrast 0,387) |
| Reihenfolge je Arm | aufsteigender Index | Isaac Lab ordnet nach Baumtiefe, und die Tiefe ist genau die Kette Schulter-Pitch → -Roll → -Yaw → Ellenbogen → Handgelenk-Roll → -Pitch → -Yaw — dieselbe Reihenfolge wie im echten Datensatz. 13/14 Wertebereiche liegen im echten Bereich (zusammenhängende Annahme `15:29`: nur 8/14) |
| Regler-Zeitversatz | **4 Frames = 80 ms** | RMSE-Minimum, gemessen nur an den Händen (dort ist die Aktion Gelenkraum) |
| Güte des Arm-Ersatzes | **0,0238 rad = 1,4°** | `hand_joint[t+4]` gegen das echte Handkommando — dort liegen beide vor |
| Handblöcke | links `j29…j35`, rechts `j36…j42` | RMSE 0,008…0,049 rad, Abstand zum Zweitbesten 3,4…15× |
| Vorzeichenkonvention der Hände | wie die offizielle DEX3-URDF, **nicht gespiegelt** | Grenzverletzung 0,0000 rad gegen die eigene Seite, 0,74 (links) / 0,14 (rechts) gegen die gespiegelte |
| **Reihenfolge innerhalb einer Hand** | `left_dex3` → `[31, 37, 41, 30, 36, 29, 35]`, `right_dex3` → `[34, 40, 42, 32, 38, 33, 39]` | **nicht** aus den Daten, sondern aus der Isaac-Gelenktabelle des eigenen USD-Assets (§ 5). Gegengeprüft: alle 14 passen in die USD-Grenzen genau ihres Gelenks, größte Verletzung 0,0005 rad, sechs davon einseitig begrenzt |

### 3.1 Warum die Fingerreihenfolge nicht aus den Daten kommt

Der Vollständigkeit halber, weil der Weg naheliegt und in eine Sackgasse führt: **aus dem
Datensatz allein** ist die Reihenfolge nicht bestimmbar. Drei Verfahren wurden probiert, keines
trennt. Aufgelöst wurde sie am Ende aus einer ganz anderen Quelle — § 5.

1. **URDF-Gelenkgrenzen.** Sie schließen viel aus, aber nicht genug: **336** (links) bzw. **144**
   (rechts) der 5040 Reihenfolgen bleiben zulässig. Immerhin fällt eine Grobstruktur ab — rechts
   ist der Daumen `{j36, j40, j42}`, Zeige- und Mittelfinger sind `{j37, j38, j39, j41}`.
2. **Korrelationsstruktur der Finger.** Abstand zwischen bester und zweitbester Permutation:
   0,003 von etwa 4. Die Greifstrategie der synth-Aufnahmen unterscheidet sich zu stark von der
   echten, als dass der Vergleich etwas trüge.
3. **Vorwärtskinematik.** Der naheliegendste Weg — Fingerkuppen ausrechnen und die Reihenfolge
   behalten, die eine plausible Greifgeometrie ergibt. Er wurde **am echten Datensatz geprüft, wo
   die Wahrheit bekannt ist**
   ([`dex3_fingerorder_probe.py`](../../Training/scripts/dex3_fingerorder_probe.py)):

   | Filterstufe | links | rechts |
   |---|---|---|
   | URDF-Grenzen (Toleranz 0,40 rad) | 144 von 5040 | 72 von 5040 |
   | + keine Fingerdurchdringung | 130 | 70 |
   | + plausible Öffnung und Schließung | **49** | **34** |
   | wahre Reihenfolge dabei? | ja — ununterscheidbar von 48 weiteren | ja — von 33 weiteren |

   **Das Verfahren gewinnt die bekannte Wahrheit nicht zurück.** Der Grund ist kinematisch und
   nicht behebbar: Zeige- und Mittelfingerkette der DEX3 sind bis auf ±2,85 cm Palm-Versatz
   identisch, ihre Kuppendistanz ist für jede Gelenkbelegung √(5,70 cm² + Δ²) ≥ 5,70 cm. Die
   Geometrie trägt die Information nicht.

   Nebenbefund, der die Toleranz erklärt: der echte Datensatz **überschreitet seine eigenen
   URDF-Grenzen um bis zu 0,34 rad**. Mit einer engeren Toleranz fällt die wahre Reihenfolge
   schon durch das erste Sieb.

Eine falsche Fingerzuordnung wäre hier der teuerste denkbare Fehler: die Fingerspanne ist die
Messgröße dieses Projekts, und 10 % der Trainingsstichproben trügen dann widersprüchliche
Fingerbefehle. Deshalb verweigert `inspect` ohne `--joint-names` die Freigabe, und `convert`
läuft nicht auf einem nicht freigegebenen Mapping — auch jetzt noch, wo die Antwort vorliegt:
sie kommt als *Eingabe* herein und wird gegengeprüft, nicht fest verdrahtet.

---

## 4. Was der Konverter tut

* **State:** 28 der 43 Vollkörper-Gelenke, in der Achsenreihenfolge des echten Datensatzes.
* **Arm-Aktion:** Ersatz aus den **gemessenen** Armgelenken bei t + 4 Frames. Das nicht
  aufgezeichnete Gelenk-Kommando lässt sich nicht rekonstruieren; der Ersatz ist an den Händen
  gegen das echte Kommando geprüft und trifft es auf 1,4°. Für Behaviour Cloning ist
  „reproduziere die Bewegung, die stattgefunden hat" ohnehin das sinnvolle Ziel — derselbe
  Gedanke wie beim eigenen Renderer, der den in der Sim *erreichten* Zustand als
  `observation.state` schreibt. Die letzten 4 Quellframes je Episode entfallen dadurch.
* **Hand-Aktion:** unverändert das echte Kommando aus `action[14:28]`.
* **Bildrate:** zeittreues Resampling 50 → 30 fps. Ausgabeframe *k* nimmt Quellframe
  `round(k · 50/30)`; Parquet und alle vier Videos werden mit **denselben** Indizes gezogen.
* **Metadaten:** v2.1-`info.json`, `tasks.jsonl` mit dem Sprachbefehl des echten Datensatzes
  (damit die Spracheingabe in beiden Domänen identisch ist), `modality.json` byteidentisch
  übernommen. Die zweite „Aufgabe" `valid` (`human.validity`) wird verworfen.
* **Split:** `{"train": "0:16"}` — alle Episoden ins Training, siehe § 7.

**Abnahme des Werkzeugs** (gegen einen künstlichen Quelldatensatz mit bekannter Wahrheit,
2026-08-28): Zuordnung und Zeitversatz exakt wiedergefunden; nach dem Umschreiben zeigen Bild,
State, Aktion und Zeitstempel Frame für Frame auf denselben Quellframe (Bildmarken dekodiert,
`max|ΔState| = 0`); mehrdeutige Zuordnung, doppelt vergebene Gelenke und ein zu kurzes Video
brechen ab bzw. kürzen konsistent; der Permutationsdetektor findet einen künstlich vertauschten
`Index`/`Middle`-Block und schlägt bei sauberen Daten nicht an.

---

## 5. Die Auflösung: die eigene Sim weiß es

Der Datensatz wurde **in dieser Umgebung** aufgenommen — mit demselben USD-Asset
(`data/g1_dex3.usd`). Damit ist die Gelenkreihenfolge kein Rätsel, sondern eine Eigenschaft
des Assets, und Isaac Lab druckt sie beim Spawn als Tabelle „Simulation Joint Information".
Sie steht in jedem Sim-Log dieses Projekts, z. B.
`Simulation/runs/20260808/04/cams-20260808-130340.log`, und liegt jetzt ausgelesen unter
[`Simulation/g1_dex3_sim/isaac_joint_order.txt`](../../Simulation/g1_dex3_sim/isaac_joint_order.txt).

Sie ist eine **andere** als die des echten Datensatzes: Isaac sortiert nach Baumtiefe und
verschachtelt links/rechts, Beine und Hände liegen dazwischen.

```
 0 left_hip_pitch    1 right_hip_pitch    2 waist_yaw
 3 left_hip_roll     4 right_hip_roll     5 waist_roll
 6 left_hip_yaw      7 right_hip_yaw      8 waist_pitch
 9 left_knee        10 right_knee        11 left_shoulder_pitch  12 right_shoulder_pitch
13 left_ankle_pitch 14 right_ankle_pitch 15 left_shoulder_roll   16 right_shoulder_roll
17 left_ankle_roll  18 right_ankle_roll  19 left_shoulder_yaw    20 right_shoulder_yaw
21 left_elbow       22 right_elbow       23 left_wrist_roll      24 right_wrist_roll
25 left_wrist_pitch 26 right_wrist_pitch 27 left_wrist_yaw       28 right_wrist_yaw
29 left_hand_index_0   30 left_hand_middle_0   31 left_hand_thumb_0
32 right_hand_index_0  33 right_hand_middle_0  34 right_hand_thumb_0
35 left_hand_index_1   36 left_hand_middle_1   37 left_hand_thumb_1
38 right_hand_index_1  39 right_hand_middle_1  40 right_hand_thumb_1
41 left_hand_thumb_2   42 right_hand_thumb_2
```

**Die Hände sind links/rechts verschachtelt.** Die Schlüssel `left_hand`/`right_hand` in der
`modality.json` des synth-Datensatzes sind deshalb irreführend: sie teilen den
Isaac-geordneten 14er-Vektor stumpf in 7 + 7, und der erste Block enthält bereits drei
Gelenke der *rechten* Hand. Wer diesen Namen geglaubt hätte, hätte die Finger vertauscht.

### 5.1 Warum das keine Annahme, sondern eine Prüfung ist

Die Liste kommt von außen — sie wird deshalb gegen die Daten gehalten, nicht geglaubt:

| Prüfung | Ergebnis |
|---|---|
| Passt jeder synth-Wertebereich in die USD-Grenze **genau des Gelenks**, das die Tabelle dort nennt? | ja, alle 14 Handgelenke; größte Verletzung **0,0005 rad** |
| Wie zufällig ist das? | Sechs der 14 haben **einseitige** Grenzen (links index/middle nur negativ, rechts nur positiv, `thumb_2` seitenverkehrt) — die synth-Daten treffen jedes Vorzeichen |
| Deckt sich die Armzuordnung mit der unabhängigen FK-Messung aus § 3? | **identisch**, alle 14 Indizes |
| Deckt sich der Handblock mit der RMSE-Messung aus § 3? | **identisch**, dieselbe Menge `29…42` |

Daraus folgt der 28er-Indexplan in der Achsenreihenfolge des echten Datensatzes:

```
left_arm    [11, 15, 19, 21, 23, 25, 27]
right_arm   [12, 16, 20, 22, 24, 26, 28]
left_dex3   [31, 37, 41, 30, 36, 29, 35]     thumb0,1,2  middle0,1  index0,1
right_dex3  [34, 40, 42, 32, 38, 33, 39]     thumb0,1,2  index0,1   middle0,1
```

`inspect` rechnet ihn selbst aus der Namensliste aus und führt die Prüfungen der Tabelle
oben durch — er ist nirgends fest verdrahtet.

> **Wenn der Datensatz aus einer FREMDEN Umgebung käme**, gälte das alles nicht. Dann bliebe
> nur, die Namensliste beim Ersteller zu erfragen (`env.scene["robot"].joint_names`) und sie
> genauso durch `--joint-names` zu schicken. Die Prüfungen greifen unverändert.

---

## 6. Ablauf

### 6.1 Lokal

Ephemere Umgebung, damit nichts im Repo-venv landet. Als **Funktion**, nicht als Variable — zsh
splittet ein `$VAR` nicht in Wörter auf und würde die ganze Zeile als einen Kommandonamen lesen:

```bash
export HF_TOKEN=hf_...        # Zugriff auf das GATED Repo muss freigeschaltet sein

harm() {
  uv run --no-project --with numpy --with pandas --with pyarrow \
         --with huggingface_hub --with imageio --with imageio-ffmpeg \
         python Training/scripts/harmonize_synth_dataset.py "$@"
}

# 1. Prüfen und Zuordnung belegen (~4 MB, keine Videos). Ohne --joint-names bricht es
#    bei der Fingerreihenfolge ab — das ist gewollt, nicht kaputt (§ 3.1).
harm inspect --source Fichtl00/Cube_Stacking_synth --work-dir data/cotrain_synth \
     --joint-names Simulation/g1_dex3_sim/isaac_joint_order.txt

# 2. Umschreiben (~150 MB Videodownload)
harm convert --work-dir data/cotrain_synth --out data/cube_stacking_synth_v21 \
     --modality-json app/Groot-1.6/examples/G1_DEX3/modality_4cam.json

# 3. Gegenprüfen (Videoframes gegen Parquetzeilen inbegriffen)
harm verify --dataset data/cube_stacking_synth_v21 \
     --modality-json app/Groot-1.6/examples/G1_DEX3/modality_4cam.json

# 4. Auf den Cluster schieben
rsync -avz --progress data/cube_stacking_synth_v21/ \
  <user>@transfer.hpc.gwdg.de:/mnt/vast-kisski/projects/kisski-humrob/data/cotrain/cube_stacking_synth/
```

Die Tabellen aus Schritt 1 gehören ins Laufprotokoll. `harmonize_report.json` im Zielverzeichnis
hält Zuordnung, Achsennamen, Ersatzverfahren, Zeitversatz und Framezahlen je Episode fest.

### 6.2 KISSKI

Die Compute-Nodes haben **kein Internet** — der Datensatz muss vor dem Job auf dem
Projektspeicher liegen, sonst läuft `run_finetuning_cotrain.sh` in einen `snapshot_download`,
der dort scheitert.

```bash
# Vorabtest, spart einen toten Job
P=/mnt/vast-kisski/projects/kisski-humrob
cmp $P/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset/meta/modality.json \
    $P/data/cotrain/cube_stacking_synth/meta/modality.json && echo "modality.json identisch"

cd $P/repo && mkdir -p logs
USE_COTRAIN=1 \
COTRAIN_DATASET_PATH=/data/cotrain/cube_stacking_synth \
COTRAIN_MIX_RATIO=0.10 \
TRAIN_TEST_SPLIT=1 TRAIN_SPLIT_RATIO=0.8 \
OUTPUT_DIR=/data/g1_dex3_finetune/blockstacking_cotrain_synth \
EXPERIMENT_NAME=g1_dex3_blockstacking_cotrain_synth_v1 \
  sbatch --export=ALL Training/kisski_submit.sh
```

Der Namespace ist bewusst neu: der Fork ruft `resume_from_checkpoint=True` fest verdrahtet auf
([`lib_resume_guard.sh`](../../Training/scripts/lib_resume_guard.sh)).

---

## 7. Warum diese Parameter

### Mischungsverhältnis 0,10

`COTRAIN_MIX_RATIO` ist der Anteil der **gerenderten** Stichproben, nicht ihr Größenverhältnis —
`ShardedMixtureDataset` normiert die Gewichte und rechnet die Shard-Größen heraus. Bei 44 000
Schritten × Batch 32 = 1,41 Mio. Stichproben und rund 3 200 synth-Frames (5 325 nach dem
Resampling auf 30 fps, minus 4 Frames je Episode für den Arm-Ersatz):

| `mix_ratio` | synth-Stichproben | Wiederholungen je synth-Frame | Epochen auf den echten Daten |
|---|---|---|---|
| natürlich (1,4 %) | 20 k | 6 | 5,7 |
| **0,10** | 141 k | **44** | 5,7 |
| 0,25 (Default der co-training.md) | 352 k | 110 | 5,3 |

Der Default 0,25 stammt aus [co-training.md](co-training.md) und war für **60 gerenderte
Episoden** (~56 k Frames) gerechnet — 17-mal mehr Material. Hier führte er auf 110 Durchläufe
durch dieselben 16 Episoden, mit aufgetautem Vision-Encoder. Das ist die Konstellation, in der
der Encoder 16 Szenen auswendig lernt statt Sim-Invarianz. Erweist sich der synth-Anteil als zu
schwach, ist 0,15 der nächste Schritt — nicht 0,25.

### Split nur auf den echten Episoden

`TRAIN_TEST_SPLIT=1` wirkt in `run_finetuning_cotrain.sh` **nur** auf den echten Datensatz:

* Die entscheidende Validierungszahl ist die MSE auf zurückgehaltenen **echten** Episoden — nur
  die ist gegen Lauf 3 (0,00716) vergleichbar.
* 20 % von 16 Episoden wären 3 Test-Episoden. Daraus lässt sich keine belastbare Zahl ziehen, und
  sie kosten 20 % eines ohnehin winzigen Satzes.
* Ob der synth-Anteil gewirkt hat, misst nicht die Open-Loop-MSE, sondern das `span`-Gate.

---

## 8. Was den Lauf beurteilt

Vorab festgelegt, damit die Auswertung hinterher nicht die Kriterien sucht:

| Frage | Messung | Bezugspunkt |
|---|---|---|
| Hat der synth-Anteil der **Realdomäne** geschadet? | `checkpoint_sweep.py`, MSE auf den zurückgehaltenen echten Episoden | Lauf 3: 0,00716 |
| Hat er den **Domain-Gap** verkleinert? | `server_rl_run.sh span` — kommandierte Fingerspanne gegen Demonstration | Lauf 34: 27,6 % (Lauf 31: 20,5 %) |
| Kopfzahl | `server_rl_run.sh eval` — `lifted` / Erfolg im Closed-Loop | bisher 0/20 |

Und in dieser Reihenfolge: Die Checkpoint-Auswahl kommt **vor** jeder Sim-Zahl. Lauf 3 hat
gezeigt, dass der letzte Checkpoint 25 % schlechter sein kann als der beste
([lauf3-vision-split-auswertung.md](../ergebnisse/lauf3-vision-split-auswertung.md)).

---

## 9. Offene Punkte

* **Die Episoden sind 4,7-mal kürzer als die echten.** 333 Frames bei 50 fps = **6,7 s** gegen
  934 Frames bei 30 fps = **31,1 s** im echten Datensatz, bei derselben Aufgabe. Zwei Lesarten:
  Teleoperation in der Sim ist schneller als am echten Roboter, oder die Episoden decken die
  Aufgabe nicht vollständig ab. **Vor dem Trainingslauf ein bis zwei `cam_left_high`-Videos
  ansehen** und im Laufprotokoll festhalten, was darauf zu sehen ist. Decken sie nur einen
  Ausschnitt ab, lernt das Co-Training ein Teilverhalten, und das Mischungsverhältnis aus § 7
  gehört nach unten.
* **Welcher Simulator?** Nicht dokumentiert. Bestimmt, wie nah die synth-Bilder an unserer
  Isaac-Lab-Zieldomäne liegen (§ 1).
* **Quaternionsreihenfolge.** Ob `action[3:7]` als (w,x,y,z) oder (x,y,z,w) abgelegt ist, spielt
  für die Harmonisierung keine Rolle — die Armaktion wird ohnehin ersetzt. Für eine spätere
  Auswertung der EEF-Bahnen wäre es zu klären.
* **16 Episoden bleiben 16 Episoden.** Kann der Ersteller nachlegen, ist mehr Material der
  wirksamste einzelne Hebel an dieser Stelle.
