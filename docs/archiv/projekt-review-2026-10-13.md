# Projekt-Review 2026-10-13

> **TL;DR:** Zweite Gesamtdurchsicht des Repos (Branch `training-luca-IKR-IS6.0`, nach dem
> Cosmos- und dem N1.7-Merge): Fehler im Code, Doku-Widersprüche, methodische Schwächen und
> Wartbarkeit, mit Priorisierung und empfohlener Reihenfolge. Reiner Befundbericht — außer den
> zwei Wartungspunkten in §0 ist nichts korrigiert. Den Vorgänger mit inzwischen behobenen
> Befunden enthält [projekt-review-2026-10-11.md](projekt-review-2026-10-11.md); den Stand der
> Umsetzung je Befund hält [§9](#9-stand-der-umsetzung-2026-10-13) fest; laufende
> Prioritäten pflegt [`next-steps.md`](../../next-steps.md).

> **Archiviert am 2026-10-29**, Befunde umgesetzt bzw. in [next-steps.md](../../next-steps.md) übernommen; Umsetzungsstand in §9.

## Inhalt

- [0. Vorgehen, Belegstatus, in dieser Sitzung erledigt](#0-vorgehen-belegstatus-in-dieser-sitzung-erledigt)
- [1. Die zehn wichtigsten Punkte](#1-die-zehn-wichtigsten-punkte)
- [2. Methodik und Experimente](#2-methodik-und-experimente)
- [3. Code: Training](#3-code-training)
- [4. Code: Simulation](#4-code-simulation)
- [5. Code: Augmentation, tools/, Forks](#5-code-augmentation-tools-forks)
- [6. Doku-Konsistenz](#6-doku-konsistenz)
- [7. Repo-Struktur und Wartbarkeit](#7-repo-struktur-und-wartbarkeit)
- [8. Empfohlene Reihenfolge](#8-empfohlene-reihenfolge)
- [9. Stand der Umsetzung (2026-10-13)](#9-stand-der-umsetzung-2026-10-13)

## 0. Vorgehen, Belegstatus, in dieser Sitzung erledigt

Sieben parallele, rein lesende Prüfungen: Training/, Simulation/, Augmentation+tools/, Forks
(`app/Groot-1.6`, `app/Groot-1.7`), Doku-Konsistenz, Methodik, Repo-Struktur. Danach
Stichproben im Code.

- **[nachgeprüft]**: im Code bzw. in der Doku selbst gesehen.
- **[Teilprüfung]**: von der Teilprüfung mit Datei:Zeile belegt, nicht einzeln nachgeprüft.

**Automatische Checks:** `tools/gen_docs.sh` ohne Abweichung (509 Parameter, 5 bewusst, 1
ungeprüft: `DR_ENABLED`), `tools/check_tldr.sh` 201/201, `tools/test_menu.sh` 81/81, `bash -n`
und Python-AST auf allen Projektdateien sauber. **shellcheck und ruff sind auf dem Host nicht
installiert**, also lief kein Lint.

**In dieser Sitzung erledigt** (die beiden Wartungspunkte aus CLAUDE.md, siehe
[historie.md](../historie.md)):

1. `setup_and_train_container_build.sh` war im Code schon seit `a8c8c66` korrigiert, es baut
   direkt per `docker build`. Nur der CLAUDE.md-Eintrag war veraltet.
2. Der Submodul-Pointer `app/Groot-1.6` zeigt jetzt auf `9508b49`, wie die Dockerfiles.
   `26ad1f7` war ein nie gepushter Doku-Commit (`FINETUNING_GUIDE.md`) und ist lokal als Tag
   `archive/unpushed-26ad1f7-guide-params` gesichert.

## 1. Die zehn wichtigsten Punkte

| # | Punkt | Bereich | Schwere |
|---|---|---|---|
| 1 | Der Closed-Loop-Misserfolg ist **nicht identifiziert**: „Domain-Gap bestätigt“ vergleicht Bedingungen, die sich in ≥ 6 Faktoren unterscheiden. Billige trennende Experimente fehlen (§2.3) | Methodik | kritisch |
| 2 | `USE_COTRAIN=1` trainiert über Entrypoint und KISSKI **ohne Held-out-Split**: `Dockerfile` und `entrypoint.sh` setzen `TRAIN_TEST_SPLIT=0` vor dem `:-1` im Cotrain-Skript | Training | hoch |
| 3 | Eval: Ein `get_action`-Timeout macht den ZMQ-REQ-Socket unbrauchbar. Alle Folge-Episoden brechen still ab und erscheinen als plausible 0/N | Simulation | hoch |
| 4 | RL/FPO: `old_logp` und `new_logp` nutzen verschiedene Rauschziehungen, die Ratio ist Rauschen. Dazu setzt `env.reset()` alle Envs zurück, und die Erfolgsmetrik ist praktisch immer 0 | Simulation | hoch |
| 5 | `lifted` (Schwerpunkt-Δz > 2 cm) misst teilweise Kippen. Die Fingerspanne sättigt bei 2,094 rad. Beide sind trotzdem die Gates für Co-Training und RL | Methodik | hoch |
| 6 | KISSKI: Das Menü bietet `kisski-h100` an, das Skript fordert aber fest `-G A100:4`. `NUM_GPUS` < 4 belegt trotzdem 4 GPUs | Training | mittel |
| 7 | Tokens landen in `ps`/`/proc/*/cmdline` (`-e HF_TOKEN=…`, `--env`, `wandb login KEY`), und `--dry-run` druckt sie aus | Training | mittel |
| 8 | Die N1.7-Doku widerspricht sich: „kein voller Lauf / nichts auf KISSKI“ steht in CLAUDE.md, README, anleitung, env-vars und docs/README, obwohl Lauf 4 fertig ist. Zudem steht „bester Checkpoint 30000“ in README und train-test-split | Doku | hoch |
| 9 | `graphify-out/` ist getrackt (1250 Dateien, ~120 MB, `.git` = 829 MB), obwohl `.gitignore` „nicht tracken“ sagt. Dazu USD-Assets ohne LFS | Repo | hoch |
| 10 | Kein CI. Die guten Checks (`gen_docs`, `check_tldr`, `test_menu`) laufen nur von Hand, ruff und shellcheck gar nicht | Repo | mittel |

## 2. Methodik und Experimente

### 2.1 Befunde

**M1. Die Ursache des Closed-Loop-Versagens ist nicht identifiziert (kritisch).** Der
`span`-Test läuft teacher-forced: jeder Chunk bekommt echten Zustand und echtes Bild
(`Simulation/scripts/finger_span_openloop.py:132-177`). Der Closed Loop unterscheidet sich
davon nicht nur im Bild:

- Sim-Zustand aus der eigenen Trajektorie, also Kovariaten-Shift mit Fehlerakkumulation
- falscher Prompt bis 2026-10-13
- Würfellagen außerhalb der Trainingsverteilung (bis `EVAL_LAYOUT`)
- 28,57 Hz statt 30 Hz
- Exec-Horizon 8 statt 16
- Kopfkamera folgt dem korrigierten Becken nicht (Review 10-11 §3.1–3.4)

Die Formulierung „Domain-Gap bestätigt“ (ab Lauf 30) ist deshalb zu stark. Der Tauschtest steht
nur als „optional“ in `next-steps.md`. [Teilprüfung]

**M2. Regeln wurden nach den Daten ausgelegt.** Die vorregistrierte Regel wurde mit Lauf 30
geschlossen, der nur 2 Episoden hatte (`diagnose-chronik.md:1327 ff.`). Lauf 29 lag unter der
eigenen 0,5-Schwelle und wurde als „dritter Fall“ umgedeutet. Sein Hub von 7,9 cm entstand durch
einen in die Finger teleportierten Würfel. Für die Arbeit sollte das offen als explorativ
ausgewiesen werden. [Teilprüfung]

**M3. Die Hilfsmetriken tragen weniger, als die Doku ihnen zuschreibt.**

- *Fingerspanne*:
  - Sie ist max−min eines Gelenks über die Episode und enthält weder Timing noch Ort.
  - Sie sättigt am De-Normalisierungsmaximum von 2,0944 rad. Das „Verhältnis 1,00“ in Lauf 32
    ist ein Deckeneffekt.
  - Lauf 54 zeigt 92 % Spanne durch Auf-Zu in der Luft.
  - Trotzdem ist die Spanne das primäre Gate in `docs/training/co-training.md:429`.
- *`lifted`*:
  - Der Schwerpunkt eines 5-cm-Würfels steigt beim Kippen auf die Kante um ≈ 1,0 cm, auf die
    Ecke um ≈ 1,8 cm. Die Schwelle liegt bei 2 cm.
  - Aussagen wie „0,3 cm bis zur Schwelle“ (`next-steps.md`) und die Spearman-Kopplung
    Spanne~Anhebung tragen deshalb nicht.
  - Vorschlag: Würfel-*Unterseite* > 2 cm über dem Tisch **und** Fingerkontakt.

**M4. Die Stichproben sind zu klein für Vergleiche.**

- 0/10 heißt nach Clopper-Pearson (95 %) eine wahre Erfolgsrate von 0–30,8 %, 0/20 heißt 0–16,8 %.
- Vergleiche N1.6↔N1.7 oder vor↔nach der Geometriekorrektur über `lifted` sind damit
  uninformativ.
- `DR_SEED` gibt es erst seit 2026-10-13, alle früheren Läufe sind ungeseedet.

**M5. Open-Loop-MSE ist als Auswahlkriterium nur bedingt geeignet.**

- Die MSE ist unnormiert und über 28 Dimensionen und die ganze Zeitachse gemittelt. Die
  Greifphase geht darin unter.
- Es gibt keine triviale Baseline (Zustand halten).
- Auswahl und Bericht laufen auf demselben Test-Set, mit mehreren t-Tests ohne Korrektur.
  44k vs. 30k ergibt p ≈ 0,045, was nach Holm nicht hält.
- Gegenbeleg im Projekt selbst: N1.7 ckpt15000→44000 senkt die Test-MSE um 33 %, aber die
  Sim-Spanne fällt von 92 % auf 26 %. Open Loop sagt den Closed Loop hier nicht vorher. Diese
  Dissoziation ist selbst ein berichtenswerter Befund.

**M6. Die Domain-Gap-Messung ist schwach.**

- Sie stützt sich auf 1 Frame pro Kamera in der Reset-Pose, nutzt SigLIP statt des
  Policy-Encoders und stammt aus der Zeit vor der Geometriekorrektur.
- Die Grundlinie „real↔real, *andere* Kamera“ ist keine echte Grundlinie.
- Sim→Real (0,22) liegt sogar *unter* real↔real (0,27). Das steht in Spannung zur Hauptthese.

**M7. Die Geometrie-Rekonstruktion ist zirkulär.**

- Basis und Würfel wurden über dieselbe Kopfkamera geschätzt, bei n = 8.
- Der Restfehler Hand→Würfel liegt bei 4,5 ± 1,7 cm, bei 5 cm Würfelkante.
- Ob die *Demonstration selbst* in der Sim greift, wurde nie gemessen.

### 2.2 Zahlen- und Aussage-Inkonsistenzen

| Aussage | Wo | Richtig |
|---|---|---|
| „0/20, Lauf 30“ | `weiterfuehrend/reinforcement-learning-plan.md:80`, `ergebnisse/domain-gap-analyse.md:93` | 0/20 ist **Lauf 24**, Lauf 30 hatte 2 Episoden [nachgeprüft] |
| Task im Training „stack the blocks“ | `ergebnisse/lauf4-n17-vergleich.md:13` | Der Datensatz kennt nur „stack three block“ [nachgeprüft] |
| „RMSE 0,085 in normierten Aktionseinheiten“ | `lauf3-vision-split-auswertung.md:139` | Der Fork rechnet unnormiert, also 0,085 rad |
| „Da `tune_visual = false`, sind die Gewichte identisch“ | `domain-gap-analyse.md:22` | Widerspricht der korrigierten Fragestellung darüber |
| „Neumessung mit korrigierter Szene steht aus“ | `next-steps.md:44`, CLAUDE.md, `diagnose-chronik.md:16` | Die Läufe 54/55 (N1.7) liefen danach. Offen ist das nur für N1.6. Umgekehrt vergleicht die Chronik 54/55 ohne Hinweis mit Lauf 34 (alte Geometrie) |
| Held-out-`span` offen | CLAUDE.md, `next-steps.md:79` | Für N1.7 gibt es ihn schon (`diagnose-chronik.md:2203`) |
| Neumessung mit `vision-v2-30000` | `next-steps.md:80` | Der validierte Checkpoint ist jetzt 44000 |
| TL;DR „belegt Overfitting“, §8.2 „MAX_STEPS=30000 genügt“ | `lauf3-vision-split-auswertung.md:4, 288` | Vom eigenen Nachtrag widerlegt |
| „27,6 % vs. 100 %, ein Drittel des Weges“ | `next-steps.md:225` | Verschiedene Nenner (Konstante 2,094 rad vs. gesättigtes Verhältnis) |

### 2.3 Empfohlene Experimente (nach Kosten/Nutzen)

1. **Orakel-Replay ohne Modell (~1 h).**
   - Ground-Truth-Aktionen der Held-out-Episoden mit `EVAL_LAYOUT` (inkl. Gierwinkel) bei 30 Hz
     abspielen.
   - Messen: `lifted`/`stacked` und den Tracking-Fehler Sim-Zustand↔aufgezeichneter Zustand.
   - Ergebnis ist die **obere Schranke**: Greift schon die Demo nicht, ist die Sim der Blocker,
     und jedes Policy-Training geht ins Leere.
2. **Teacher-forced Open Loop auf Sim-Bildern (~2 h).**
   - `render_cotrain_dataset.py` auf den Test-Episoden laufen lassen, danach
     `checkpoint_sweep.py`/`span` auf dem gerenderten Satz.
   - Gleicher Zustand, gleiche Aktion, nur das Bild ist anders. Das ist der eigentliche Test der
     Domain-Gap-Hypothese.
3. **Bild-Ablation im Open Loop (~30 min).**
   - Bilder schwärzen, mischen oder kameraweise tauschen.
   - Ändert sich wenig, nutzt die Policy vor allem den Zustand. Dann ist die Zustandsverteilung
     im Closed Loop der Hauptkandidat, nicht das Bild.
4. **Tauschtest im Closed Loop.** Echte Bilder zeitsynchron einspeisen, während die Sim
   ausführt.
5. **Saubere Baseline-Neumessung.**
   - Alle Fixes zusammen: Prompt, Held-out-Layout, 30 Hz, Seed, Kopfkamera-Entscheid.
   - N1.6@44k und N1.7@44k mit je 20 Episoden, mit Konfidenzintervall und kipp-robustem `lifted`.
6. **Open-Loop-Protokoll härten.**
   - Triviale Baselines, MSE getrennt nach Arm und Hand, Greifphase gewichtet.
   - Exec-Horizon 8, alle 61 Test-Episoden, Bootstrap-KI, Holm-Korrektur.
7. **Erst danach Co-Training und RL.**
   - Neu rendern erst nach Klärung von Kamera und Becken und nach Experiment 1.
   - Cosmos erzeugt keine Sim-Optik, wirkt also höchstens indirekt.
   - RL erst ab `lifted` > 0.

**Einschätzung:** Co-Training, Cosmos und RL zielen alle auf den visuellen Gap. Nach heutiger
Beleglage ist das nur einer von mehreren gleich plausiblen Kandidaten. Die Experimente 1–4 sind
billiger und trennen die Kandidaten.

### 2.4 Für die schriftliche Arbeit

- Kopfzahlen mit KI angeben, etwa „0/10 (95 %-KI 0–31 %)“.
- Eine Tabelle aller Closed-Loop-Läufe mit Bedingungen: Geometrie vor/nach 24.08., Prompt,
  Layout, Hz, Seed, Checkpoint, n. Läufe ≤ 34 als „unter bekannten Messfehlern“ markieren.
- „Bestätigt/ausgeschlossen“ ersetzen durch „vereinbar mit …; Alternativen nicht
  ausgeschlossen“. Konfirmatorisches und Exploratives trennen.
- Die widerlegte U-Kurve (n = 6 kurze Episoden) als methodische Lektion zeigen, nicht
  verschweigen.
- Den Beitrag ehrlich rahmen: eine reproduzierbare Pipeline, eine Kette gefundener Messfehler
  (Kameras, Fingervorzeichen, Becken, Prompt, Layout), ein sauber dokumentiertes Negativergebnis.
- N1.6 und N1.7 nur im Open Loop vergleichen.

## 3. Code: Training

**T1 — hoch [nachgeprüft]: Der Co-Training-Split ist über Entrypoint und KISSKI aus.**

- `run_finetuning_cotrain.sh:81` setzt `TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-1}"` („NICHT
  optional“, so steht es auch in env-vars.md und co-training.md).
- Aber `Training/Dockerfile:192` setzt `TRAIN_TEST_SPLIT=0`, `entrypoint.sh:110` `:-0`, und
  `kisski_submit.sh:185` reicht `0` weiter.
- Folge: `checkpoint_sweep.py` bricht ab oder misst Trainings-MSE. Sicher sind nur
  `kisski_chain.sh` und der direkte Aufruf.
- Fix: Im Entrypoint bei `USE_COTRAIN=1` den Default `1` setzen und das ENV aus dem Dockerfile
  entfernen.

**T2 — mittel [nachgeprüft]: Partition und GRES passen nicht zusammen.**

- `kisski_submit.sh:55-56` hat `-p kisski` und `-G A100:4` fest eingetragen.
- `kisski_menu.sh:74` ändert nur `--partition`. `kisski-h100` mit `A100:4` wird abgelehnt oder
  bleibt ewig in der Queue. `NUM_GPUS=1|2` belegt trotzdem 4 GPUs.
- `kisski-hpc.md` nennt `-G A100:1`.
- Fix: Das Menü gibt `--gpus=<typ>:$NUM_GPUS` passend zur Partition mit.

**T3 — mittel [nachgeprüft]: Der Docker-Launcher reicht `DATA_DIR` und `WANDB_DIR` des Hosts in
den Container** (`setup_and_train_dockerhub_pull.sh:399`). Ein Host-`DATA_DIR` aus `.env.local`
oder aus der KISSKI-Doku überschreibt `/data`. `run_finetuning*.sh` codieren aber `/data` fest,
die Pfade laufen also auseinander. Fix: beide aus der Liste nehmen.

**T4 — mittel [Teilprüfung]: Secrets in der Kommandozeile.**

- Betroffen: `setup_and_train_dockerhub_pull.sh:376,383,405`, `kisski_submit.sh:323,467`,
  `kisski_rl_submit.sh:137` und `wandb login --relogin "$WANDB_API_KEY"` in
  `run_finetuning*.sh:138`.
- Auf geteilten KISSKI-Knoten ist das für andere Nutzer lesbar.
- `--dry-run` druckt die Tokens im Klartext.
- Fix:
  - `-e HF_TOKEN` ohne Wert übergeben.
  - Auf KISSKI `APPTAINERENV_HF_TOKEN` nutzen.
  - `wandb login` streichen, die Umgebungsvariable reicht.
  - Im Dry-Run maskieren.
- Dazu: `GITHUB_TOKEN` aus der Klon-URL (`kisski_submit.sh:291`) landet in `.git/config`.

**T5 — niedrig: Weitere Punkte.**

- `SAVE_STEPS`/`SAVE_TOTAL_LIMIT` sind in vision und cotrain 1000/5, im Standardlauf 2000/10. Bei
  44k Steps bleiben also nur 5 Checkpoints, zu wenig für den Sweep.
- `launch_cotrain_n17.py:75` hat `cotrain_mix_ratio` default 0,5 statt 0,25 [nachgeprüft].
  Harmlos, solange das Shell-Skript den Wert setzt.
- `update_image.sh:110`: `$?` nach `if ! "$@"` ist immer 0.
- `kisski_chain.sh:123` kennt weniger `GROOT_VERSION`-Schreibweisen. Besser
  `groot_normalize_version` nutzen.
- `entrypoint.sh` hält einen abgebrochenen Download für vollständig (nur „nicht leer“ wird
  geprüft) und lädt das N1.7-Backbone nicht nach, wenn Modell und Datensatz schon da sind.
  Vorschlag: `.done`-Marker.
- `run_finetuning*.sh` codieren `/data` statt `$DATA_DIR` fest.
- `docker-compose.yml` reicht nur 7 Variablen durch (fehlen: `TUNE_VISUAL`, `TRAIN_TEST_SPLIT`,
  `USE_COTRAIN` …).
- Banner „GR00T N1.6“ fest im Pull-Launcher.
- Der GPU-Check lädt auch bei `--dry-run` ein Image.
- `checkpoint_sweep.py:52` fällt ohne `GROOT_ROOT` auf `/app/Groot-1.6` zurück, im N1.7-Image
  also falsch.

**In Ordnung:**

- Die beiden `lib_groot_version.sh` sind identisch.
- `launch_cotrain*.py` entsprechen den Fork-`launch_finetune.py` bis auf den Datensatzblock.
- `checkpoint_sweep.py`, `lib_split.sh` und der Fork-Split-Patch stimmen überein.

## 4. Code: Simulation

**S1 — hoch [nachgeprüft]: Eval-Fehler werden verschluckt.**

- `client.py:97-98`: ein REQ-Socket mit `RCVTIMEO`. Nach einem Timeout wirft jedes weitere
  `send` EFSM, der Socket wird nie neu aufgebaut.
- `run_g1_dex3_sim_eval.py:405-406` macht `print(...); break`, die Episode zählt als normales
  Ergebnis.
- Ein kalter erster Inferenzschritt über 20 s (compile/TRT) ergibt so ein plausibles 0/N.
- `ping()` fängt nur `zmq.error.Again`, der zweite Versuch scheitert an EFSM. Das verdeckt im
  Moment nur der TCP-Port-Wait im Entrypoint.
- Fix: Socket neu aufbauen, `"error": true` je Episode setzen, nach 2 Fehlern in Folge mit
  Exit ≠ 0 abbrechen und die Zusammenfassung als „N gültig / M abgebrochen“ ausgeben.

**S2 — hoch [nachgeprüft im Code, Wirkung vermutet]: FPO-Ratio aus verschiedenen Zufallsziehungen.**

- `fpo_logprob_proxy` (`rl_finetune.py:~238-250`) ruft `model.forward` auf, das Rauschen und `t`
  jedes Mal neu zieht. `old_logp` und `new_logp` sind also nicht vergleichbar, und die Ratio ist
  schon in Epoche 0 ≠ 1.
- FPO verlangt dieselben (ε, t).
- Fix: Die Ziehungen bzw. einen Generator-Seed beim Rollout speichern und im Update
  wiederverwenden, und in Epoche 0 prüfen, dass die Ratio ≈ 1,000 ist. Das ist ein billiger
  Selbsttest.

**S3 — hoch [nachgeprüft]: `rl_finetune.py:443-444` `if done.any(): env.reset()`** setzt *alle*
Envs zurück, obwohl `DirectRLEnv` fertige Envs schon automatisch zurücksetzt. Bei `num_envs > 1`
bricht das laufende Trajektorien ohne `done` ab, und GAE bootstrapt über den Sprung. Fix: den
Aufruf entfernen.

**S4 — mittel: Weitere Schwächen in RL und Eval.**

- `succ = env._episode_success.mean()` ist ≈ 0, weil `_reset_idx` das Flag schon gelöscht hat.
  Wie in der Eval `terminated & success` zählen.
- RL führt nur Chunk-Schritt 0 aus und plant jeden Schritt neu, die Eval führt 8 Schritte je
  Chunk aus. RL optimiert also ein anderes Verhalten, als gemessen wird.
- Timeouts werden als `done` ohne Truncation-Bootstrap behandelt.

**S5 — mittel: Eine veraltete `results.json` verdeckt Abstürze.**

- `entrypoint_sim.sh:~510` und `entrypoint_baseline.sh:282` prüfen nur „Datei nicht leer“.
- Der Docker-Pfad über `server_rl_run.sh` sucht den Marker `[eval] fertig.`. KISSKI und vast.ai
  tun das nicht.
- Fix: `rm -f` vor dem Start und auf den Marker prüfen. Besser: Ergebnisdatei je Lauf mit
  Zeitstempel und Checkpoint.

**S6 — mittel [nachgeprüft]: `HELD_OUT_FIRST_EPISODE = 240` ist fest eingetragen**
(`run_g1_dex3_sim_eval.py:186`). Das stimmt nur bei 300 Episoden und Ratio 0,8. Synth-Sätze oder
eine andere Ratio führen still zu falsch beschrifteten Episoden. `n_train` gehört in
`layout.json`.

**S7 — niedrig:**

- Kopfkamera und Becken (bekannt, Review 10-11 §3.4). Die Konstante `0.85` (alte Basis-z) in
  `camera_geometry.py:193` gehört nach `scene_geometry.py`.
- Veraltete Erwartungswerte und Kommentare:
  - `dump_camera_poses.py:418` (Root z 0,85)
  - `run_g1_dex3_replay.py:332` („Tisch 0.89“)
  - Eval-Banner „GR00T N1.6“ auch bei N1.7
  - tote Zeilenverweise in `entrypoint_sim.sh` und `rl_finetune.py`
- `_widen_finger_joint_limits` deckt `thumb_0`/`thumb_2` nicht ab, obwohl der Datensatz-Init dort
  Werte bis −0,71 hat. Das sollte `tipcheck` einmal prüfen.

**In Ordnung:**

- Zustands- und Aktionsaufteilung (0:7/7:14/14:21/21:28) ist gleich in `build_obs` und in den
  Modality-JSONs, die für 1.6 und 1.7 byte-identisch sind.
- Chunk 16, Exec 8.
- Das Vorzeichen ist überall konsistent (`_SIGN_FLIP_IDX = []`).
- `scene_geometry.py` ist die einzige Quelle.
- 210 Hz/7 = 30 Hz.
- Der Task-Text kommt aus dem Datensatz.
- Die N1.7-Guards greifen.
- Die Env-Defaults stimmen mit der CLAUDE.md-Tabelle überein.

## 5. Code: Augmentation, tools/, Forks

**A1 — hoch (vermutet): Die strikte Frame-Prüfung kann den ganzen Lauf verwerfen.**
`assemble_dataset.py:57-79` (`AUGMENT_STRICT_FRAME_CHECK=1`) verwirft jede Episode mit
abweichender Frame-Zahl. Cosmos arbeitet in festen Chunks. Weicht die Länge ab, endet ein
stundenlanger GPU-Lauf mit „keine Episode bestanden“. Die fps werden gar nicht verglichen.

- Fix: vorab ein 2-s-Clip mit `ffprobe`-Vergleich (`--dry-validate`).

**A2 — mittel: Die erzeugte `info.json` ist teils veraltet.** `info.json` ist eine Kopie der
Quelle:

- `total_videos`, `total_chunks` und `total_tasks` sind veraltet.
- `stats.json`, `relative_stats` und `episodes_stats` werden nicht geschrieben.

Ob der Cotrain-Loader die Statistik neu rechnet, ist ungeprüft.

**A3 — mittel: Kein Guard für die Split-Ratio.** `build_controlnet_specs.py:53-68` hat keine
Prüfung `0 < n_train < total`. Bei `AUGMENT_TRAIN_RATIO=1.0` werden die Test-Episoden augmentiert
und landen über Cotrain im Training.

**A4 — mittel: Alte Ausgaben stören einen Neulauf.**

- `AUGMENT_OUT_DIR` wird vor einem Neulauf nicht geleert. Alte Episoden über dem neuen
  `total_episodes` bleiben liegen.
- `find_produced_video` nimmt `sorted(rglob("*.mp4"))[0]` und kann so ein altes oder halbes MP4
  greifen.

**A5 — mittel [nachgeprüft]: Falscher Skriptname in Hinweistexten.**
`Augmentation/scripts/entrypoint.sh:323` nennt `./Training/setup_and_train_DockerHub-pull.sh`.
Die Datei heißt `setup_and_train_dockerhub_pull.sh`. Derselbe Fehler steht in
`setup_and_augment_DockerHub-pull.sh:8,211`.

**A6 — niedrig:**

- `download_source_dataset.sh:30` hält jedes nicht leere Verzeichnis für vollständig.
- `lib_docker_ns.sh:121` matcht auch auskommentierte Zeilen. Fix: `^[^#]*DOCKER_NAMESPACE`.
- `gen_docs.sh` legt eine vorhersagbare `/tmp`-Datei ohne `trap` an.
- `.env.local` wird gesourct, ist also Code. Das ist bewusst so, sollte aber in
  `portabilitaet.md` als Vertrauensgrenze stehen.

**Forks.**

- Die Configs für 1.6 und 1.7 sind byte-identisch. Der Split-Patch stimmt mit `lib_split.sh`
  überein.
- Risiken:
  - N1.7 hat Modell-default `action_horizon=40`, Padding/Maske auf 16 funktionieren. Einmal im
    `config.json` eines Checkpoints prüfen.
  - Die Normalisierungsstatistik wird über alle Episoden berechnet, inklusive Test. Das ist ein
    kleines Leck.
  - `HF_HUB_OFFLINE="0"` gilt im Offline-Fix `cf2032d` als wahr. Fix: `in ("1","true")`.
  - Der Split-Filter behält ohne `splits` stillschweigend alle Episoden.
  - ~~`origin/luca/g1-dex3-n17` (`efa0169`) liegt einen Commit hinter dem Pin `cf2032d`.~~
    **Korrigiert 2026-10-13:** `cf2032d` liegt auf `origin/luca/g1-dex3-n17`; der lokale
    Remote-Ref war nur veraltet. Kein Risiko wie bei 1.6.

## 6. Doku-Konsistenz

**D1 — hoch: N1.7-Status widersprüchlich.** Volle N1.7-Läufe (KISSKI 44k, Sim 54/55) sind fertig,
aber folgende Stellen sagen noch „kein voller Lauf / nichts auf KISSKI“ oder „ungetestet“:

- CLAUDE.md, Zeile der `GROOT_VERSION`-Env-Tabelle
- CLAUDE.md, Architekturbaum bei `app/Groot-1.7` („no full run yet“)
- `README.md:19` („Ungetestet: noch kein abgeschlossener N1.7-Lauf“), während README:38 auf
  Lauf 4 verlinkt
- `docs/training/anleitung.md:44`
- `docs/training/env-vars.md:49`
- `docs/README.md:100`
- `docs/weiterfuehrend/groot-n17-migration.md:12` („Stand 2026-08-19 … kein Lauf“)

Ursache: Der N1.7-Stand steht in CLAUDE.md sechsmal. Lösung: ein Status-Absatz plus Link.

**D2 — hoch: „Bester Checkpoint 30000“** steht noch in `README.md:31` (Stand 10-11) und
`docs/training/train-test-split.md:42`. Richtig ist 44000 laut der Neumessung 20×750. In Lauf 3
§8.2 und im Review 10-11 fehlt ein Hinweis „überholt“.

**D3 — mittel: Kaputte Links.**

- `docs/training/anleitung.md:539`: Anker `#checkpoint-auswahl-nach-dem-lauf-checkpoint_sweeppy`
  fehlt in env-vars.md.
- `SETUP_DOCUMENTATION.md:11`: Anker fehlt.
- `SETUP_DOCUMENTATION.md:79`: Pfad relativ zum Groot-Baum.

**D4 — mittel: Rund 70 Code-Variablen sind nirgends dokumentiert.** Darunter
nutzerrelevante: `KISSKI_PARTITION`, `KISSKI_TIME`, `RL_SHM_SIZE`, `RL_OUTPUT_DIR`, `OPTIMIZE_*`,
`LATENCY_*`, `LAYOUT_*`, `RENDER_*`, `TIPCHECK_*`, `REPLAY_*`, `SKIP_GIT_PULL`, `MASTER_PORT`.
Entweder ergänzen oder ausdrücklich auf die Menü-Specs verweisen.

**D5 — mittel: Länge.**

- CLAUDE.md hat ~57 KB und wird in jede Sitzung geladen. Projektüberblick, Status und
  weiterfuehrend-Aufzählung erzählen die N1.7-Geschichte mehrfach, das hat die Widersprüche
  erzeugt.
- Der Architekturbaum dupliziert `docs/README.md`.
- Ziel: unter ~15 KB, Status nach `next-steps.md`.
- `diagnose-chronik.md` (2233 Zeilen) und `cli-menuefuehrung.md` (1117 Zeilen) sollten nach
  Zeitraum bzw. Thema geteilt werden.

**D6 — niedrig:**

- Ob das Sim-Image mit N1.7 auf Docker Hub liegt, ist in `sim-eval-anleitung.md:76` unklar.
- Die moduswählige Lernrate (1e-4 vision/cotrain, 2e-4 Standard) fehlt in CLAUDE.md.

## 7. Repo-Struktur und Wartbarkeit

**R1 — hoch [nachgeprüft]: `graphify-out/` ist getrackt.**

- 1250 Dateien mit datierten Snapshots von je 8–13 MB. `.gitignore:107` ist auskommentiert.
- `graphify update .` lässt die Historie bei jedem Lauf wachsen. `.git` hat 829 MB.
- Fix: Zeile einkommentieren und `git rm -r --cached graphify-out`. Die graphify-Reste unter
  `latex/` ebenso.
- Optional: die Historie umschreiben (`git filter-repo`). Das braucht Absprache im Team und einen
  Force-Push.

**R2 — hoch: Kein CI.** Kein `.github/workflows`, kein `.pre-commit-config.yaml`. Vorschlag für
einen minimalen Workflow:

- `gen_docs.sh`, `check_tldr.sh`, `test_menu.sh`
- `ruff check`/`ruff format --check` auf Training/, Simulation/, Augmentation/
- `shellcheck`
- die Isaac-freien `test_*.py`
- `diff` der beiden `lib_groot_version.sh` und Pin-Abgleich zwischen Dockerfiles und
  Submodulen

Die Ruff-Konfiguration gibt es bisher nur im Fork (`app/Groot-1.6/pyproject.toml`). Ein
Root-`pyproject.toml` fehlt.

**R3 — hoch: Große Binärdateien ohne LFS.** `data/configuration/g1_dex3_base.usd` (40 MB),
`g1_gripper_base.usd` (28 MB) und weitere USDs. Dazu Lauf-Artefakte unter
`data/g1_dex3_finetune/…`. Fix: LFS oder ein HF-Artefakt mit Fetch-Skript.

**R4 — mittel: Logik mehrfach implementiert.**

- Die Split-Formel `int(total*ratio)` steht an 6 Stellen:
  - `lib_split.sh`
  - `build_controlnet_specs.py`
  - `extract_block_layout.py`
  - `reconstruct_cube_poses.py`
  - `render_cotrain_dataset.py`
  - `run_dataset_replay_videos.py`
  - dazu die Konstante 240 aus S6
- Die Commit-Pins stehen dreifach. In `Simulation/Dockerfile.standalone:102` steht der N1.6-Pin
  als nacktes Literal ohne ARG.
- Die KISSKI-Logik (SIF-Wahl, Pin-Check, Fork-Bind) steht dreifach in den SLURM-Skripten.
- Fix: ein kleines Isaac-freies `split_math.py` mit Test über alle Aufrufer, ein ARG je
  Generation und ein `kisski_common.sh`.

**R5 — mittel: Branchname fest eingetragen.** `training-luca-IKR-IS6.0` steht in
`kisski_submit.sh:120`, `setup_and_train_container_build.sh:54` und in der Doku von
`update_image.sh`. Das bricht, sobald `main` der Arbeitsbranch wird.

**R6 — mittel: Wenig Tests für eigenen Code.** Es gibt 8 `test_*.py`, alle in
`Simulation/g1_dex3_sim`. Ohne Test sind:

- `checkpoint_sweep.py`
- `harmonize_synth_dataset.py`
- `assemble_dataset.py`
- `build_controlnet_specs.py`
- `lib_split.sh`
- `lib_resume_guard.sh`

Python-Abhängigkeiten für Host-Werkzeuge sind nirgends gepinnt.

**R7 — niedrig:**

- `.gitignore` enthält Altlasten (`/proc/`, `/sys/` … und eine Whitelist für Groot-1.6 aus der
  Zeit vor dem Submodul).
- Die Benennung ist uneinheitlich: `setup_and_augment_DockerHub-pull.sh` gegenüber
  `setup_and_train_dockerhub_pull.sh`.
- Das veraltete `run_finetuning.ps1` wird weiter ins Image kopiert.
- Keine Secrets gefunden. Nur der Platzhalter in `.env.local.example`.

## 8. Empfohlene Reihenfolge

1. **Experimente vor neuen Features.** Orakel-Replay, teacher-forced Sim-Bilder, Bild-Ablation
   (§2.3, 1–3). Sie kosten wenige Stunden und entscheiden, ob Co-Training, Cosmos und RL
   überhaupt am richtigen Hebel ansetzen.
2. **Messwerkzeuge reparieren, bevor neu gemessen wird:**
   - S1 (Eval-Fehler sichtbar machen)
   - kipp-robustes `lifted`
   - S6 (`n_train` statt 240)
   - S5 (veraltete Ergebnisse)
   - Danach die saubere Baseline-Neumessung N1.6@44k / N1.7@44k mit 20 Episoden und KI.
3. **Schnelle Code-Fixes:**
   - T1 (Cotrain-Split)
   - T2 (GRES)
   - T3 (`DATA_DIR`)
   - T4 (Tokens)
   - A3 (Split-Guard)
   - A5 (Skriptname)
   - S3 (`env.reset`)
4. **Doku:** D1 und D2 auf einen Stand bringen, die Zahlen aus §2.2 korrigieren, CLAUDE.md
   verschlanken.
5. **Repo:** `graphify-out/` aus dem Tracking nehmen und ein minimales CI aufsetzen (R1, R2).
   Danach LFS, `split_math.py` und den Branchnamen (R3–R5).
6. **RL erst, wenn `lifted` > 0 ist.** Vorher S2–S4 beheben, sonst ist ein ausbleibender
   Lerneffekt nicht interpretierbar.

## 9. Stand der Umsetzung (2026-10-13)

Nachgetragen am selben Tag, nachdem die Befunde im Code und in der Doku bearbeitet wurden. Nichts
davon ist committet; die Abschnitte §1–§8 bleiben als Befundstand stehen.

- **erledigt** — umgesetzt, ohne Hardware prüfbar und geprüft (Checks, Code-Stichprobe)
- **erledigt, IKR ausstehend** — umgesetzt, die Wirkung zeigt sich erst auf dem Sim-Server
- **vorbereitet** — fertig im Arbeitsbaum, braucht noch einen Commit (Fork, Pin, Index)
- **offen** — nicht angefasst

| ID | Stand | Anmerkung |
|---|---|---|
| M1 | offen | Trennende Experimente (§2.3, 1–4) stehen aus |
| M2 | offen | Betrifft die Darstellung in der Arbeit (`latex/`) |
| M3 | erledigt, IKR ausstehend | neue Stufe `lifted_clear` (Unterseite > 2 cm + Handkontakt-Proxy ≤ 4 cm), `max_clear_lift_cm`; altes `lifted` bleibt zum Vergleich |
| M4 | erledigt, IKR ausstehend | 95-%-Clopper-Pearson-KI in Zusammenfassung und JSON (`milestone_ci95`); `DR_SEED` seit 10-13 |
| M5, M6, M7 | offen | Open-Loop-Protokoll, Domain-Gap-Messung, Geometrie-Prüfung |
| §2.2 | erledigt | alle neun Zahlen korrigiert bzw. mit „überholt“-Vermerk |
| T1 | erledigt | Co-Training-Split über Entrypoint und KISSKI an, Dockerfile-`ENV` entfernt |
| T2 | erledigt | `kisski_menu.sh` setzt GPU-Typ/-Anzahl nach Partition und `NUM_GPUS` |
| T3 | erledigt | Pull-Launcher reicht `DATA_DIR`/`WANDB_DIR` nicht mehr durch (`TRAIN_HOST_DATA_DIR`) |
| T4 | erledigt | `-e NAME` ohne Wert, `APPTAINERENV_*`, kein `wandb login <key>`, `GITHUB_TOKEN` per Credential-Helper statt in `.git/config`; Dry-Runs zeigen keine Tokens mehr |
| T5 | erledigt | `SAVE_STEPS`/`SAVE_TOTAL_LIMIT` 2000/15 überall; `cotrain_mix_ratio` 0,25; `update_image.sh`-Exit-Code; `kisski_chain.sh` nutzt `groot_normalize_version`; Download-Marker `.done`/`.incomplete`; `DATA_DIR` statt `/data`; Banner nach Generation; kein GPU-Test im Dry-Run; `checkpoint_sweep.py` findet `GROOT_ROOT` |
| S1 | erledigt, IKR ausstehend | Socket-Neuaufbau, `"error": true` je Episode, Abbruch nach 2 Fehlern in Folge, „N gültig / M abgebrochen“ |
| S2 | umgesetzt, IKR ausstehend | Seeds der FPO-Ziehungen gespeichert und wiederverwendet; Ratio-Selbsttest in Epoche 0 |
| S3 | erledigt, IKR ausstehend | globales `env.reset()` entfernt |
| S4 | umgesetzt, IKR ausstehend | Erfolg vor dem Auto-Reset gezählt, Truncation-Bootstrap, `RL_EXEC_HORIZON` |
| S5 | erledigt, IKR ausstehend | alle Entrypoints löschen `results.json` vorab und verlangen `[eval] fertig.`; Archiv je Lauf |
| S6 | erledigt, IKR ausstehend | `n_train` in `layout.json`, Rückfall mit Warnung |
| S7 | erledigt, IKR ausstehend | `CAMERA_CALIB_BASE_Z` in `scene_geometry.py`, veraltete Kommentare/Banner. **Ausgenommen** `thumb_0`/`thumb_2`: als offener Wartungspunkt in CLAUDE.md (braucht `tipcheck` auf dem IKR) |
| A1 | erledigt | Preflight-Clip vor dem Volllauf (`AUGMENT_PREFLIGHT`), Toleranz `AUGMENT_FRAME_TOLERANCE`, fps-Vergleich |
| A2 | erledigt | `stats.json` aus der Ausgabe neu berechnet (nur Train); `relative_stats` rechnet die Factory |
| A3 | erledigt | Guard `0 < n_train < total` |
| A4 | erledigt | eigener Ausgabeordner wird geleert, fremder abgelehnt; `find_produced_video` nimmt nicht mehr blind die erste Datei |
| A5 | erledigt (Teil) | Hinweistexte nennen `setup_and_train_dockerhub_pull.sh`; Verweise auf den umbenannten Augmentation-Launcher `…_dockerhub_pull.sh` angepasst (R7, `git mv` beim Commit) |
| A6 | erledigt | Download-Marker; `lib_docker_ns.sh` ignoriert Kommentarzeilen; `gen_docs.sh` mit `mktemp` + `trap`; Vertrauensgrenze `.env.local` in [portabilitaet.md](../portabilitaet.md) |
| Forks: `cf2032d` gepusht | erledigt | liegt auf `origin/luca/g1-dex3-n17` (§5 korrigiert) |
| Forks: `HF_HUB_OFFLINE`-Wahrheitswert | erledigt | Fork-Commit `288d1de` gepusht, Pin in Submodul und beiden Dockerfiles nachgezogen; wirksam mit dem nächsten N1.7-Image |
| Forks: `action_horizon` 40 vs. 16 | offen | über Masken abgefangen; `config.json` eines Checkpoints noch nicht nachgesehen |
| Forks: Normalisierungsstatistik inkl. Test, Split-Filter ohne `splits` | offen | |
| D1, D2, D3, D4, D6 | erledigt | ein N1.7-Status, 44000, Links (Linkcheck 0 kaputt), Env-Vars, Docker-Hub-Hinweis + Lernrate |
| D5 | erledigt (Teil) | CLAUDE.md 57,7 → ~13 KB; das Teilen von `diagnose-chronik.md`/`cli-menuefuehrung.md` ist zurückgestellt (eingehende §-/Anker-Verweise aus Code und `latex/`) |
| R1 | erledigt | `.gitignore` schließt `graphify-out/` aus, 1259 Dateien aus dem Index entfernt (`76f2775`); Historie unverändert (`git filter-repo` optional) |
| R2 | erledigt | CI ([`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)), Root-`pyproject.toml`; `ruff format --check` noch nicht blockierend, `Augmentation/tests/` noch nicht im CI |
| R3 | offen | LFS bzw. HF-Artefakt für die USDs |
| R4 | erledigt (Sim-Teil), IKR ausstehend | `split_math.py` als sim-seitige Quelle; Pins und KISSKI-Logik weiter mehrfach |
| R5 | erledigt | Branch aus dem Checkout, sonst `main` |
| R6 | teilweise | Tests für `eval_metrics`, `split_math`, Augmentation-Skripte; weitere offen |
| R7 | erledigt | `run_finetuning.ps1` nicht mehr im Image, `.gitignore`-Altlasten entfernt, Augmentation-Launcher umbenannt (`git mv` beim Commit) |

**Verifikation auf dem IKR-Server (ausstehend), in dieser Reihenfolge:**

1. **Eval, 2 Episoden** — `NUM_EPISODES=2 EPISODE_LENGTH_S=40 ./Simulation/server_rl_run.sh eval`:
   `[eval] fertig.`, `results.json` plus Archivdatei, `num_valid=2`, `milestone_ci95` vorhanden.
2. **Abbruchpfad** — während einer Eval den GR00T-Server beenden: Episode `"error": true`, nach
   zwei Fehlern Exit ≠ 0, Zusammenfassung „N gültig / M abgebrochen“.
3. **`lifted_clear` plausibel** — im Replay bzw. an einer Episode mit gekipptem Würfel: `lifted`
   darf anschlagen, `lifted_clear` nicht.
4. **RL-Ratio-Selbsttest** — `RL_ITERATIONS=2 RL_NUM_ENVS=2 … rl`: Log-Zeile
   `[rl] Ratio-Selbsttest` mit max|r−1| ≲ 1e-3; danach dasselbe mit `RL_EXEC_HORIZON=8`.
5. **Sim-Image neu bauen** (`update_sim_image.sh --standalone`) — die Baseline-/`g1_gripper_sim`-
   und Entrypoint-Änderungen wirken nur im neu gebauten Image; danach einmal `SIM_MODE=baseline`.

