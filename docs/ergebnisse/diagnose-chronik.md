# Diagnose-Chronik — Sim- und RL-Läufe 08–34

Chronologisches Protokoll der Diagnose-, Kalibrier- und Eval-Läufe der Isaac-Lab-Sim
(2026-08-08 bis 2026-08-14). Entstanden am 2026-08-18 durch Aufteilung von
[rl-anleitung.md](../weiterfuehrend/rl-anleitung.md), die auf 2197 Zeilen angewachsen war und
operative Anleitung mit Protokoll vermischte. Die Lauf-Nummern (`Lauf 8` … `Lauf 34`) sind der
projektweit referenzierte Zähler — andere Dokumente verlinken gezielt auf einzelne Abschnitte hier.

Für die **operative Bedienung** der Diagnose-Werkzeuge (`gap` · `eval` · `grasp` · `span`) siehe
[../weiterfuehrend/rl-anleitung.md](../weiterfuehrend/rl-anleitung.md#diagnose-werkzeuge-gap--eval--grasp--span).
Für den **aktuellen Projektstatus** siehe
[../weiterfuehrend/reinforcement-learning-plan.md](../weiterfuehrend/reinforcement-learning-plan.md)
(die eine gepflegte Statusquelle).

## Inhaltsverzeichnis der Läufe

| Lauf | Ein-Zeilen-Ergebnis |
|---|---|
| 8 | Würfel schießen beim Reset ~1 m weg (überlappende Startpositionen, PhysX-Durchdringung) — Abschnitt „Würfel landen nicht auf dem Tisch" |
| 9 | Env-Versatz-Fix verifiziert (Würfel korrekt auf dem Tisch); Kamerabilder aber weiterhin leer/weiß — Abschnitt „Würfel landen nicht auf dem Tisch" |
| 10 | Belichtungs-Sweep widerlegt Belichtung als Ursache der weißen Bilder — Abschnitt „Kamerabilder gleichmäßig weiß" |
| 11 | Domain Randomization widerlegt; 3 von 5 Kameras zeigen nur einen Grauwert (reine Dome-Farbe) — Abschnitt „Kamerabilder gleichmäßig weiß" |
| 12 | Sensorklasse (`TiledCamera`) widerlegt; Diagnostik widerspricht erstmals den Bildern selbst — Abschnitt „Kamerabilder gleichmäßig weiß" |
| 13 | USD-Stage als unabhängiger Zeuge: Kameras 95,6–136,8° falsch orientiert — Fix geschrieben — Abschnitt „Kamerabilder gleichmäßig weiß" |
| 14 | Fix bestätigt für Kopf- und Szenekamera; Wrist-Kameras wegen Namensfehler noch übersprungen — Abschnitt „Kamerabilder gleichmäßig weiß" |
| 15 | Alle 5 Kameras (20 Prims) korrekt ausgerichtet — Kameras vollständig repariert — Abschnitt „Kamerabilder gleichmäßig weiß" |
| 16 | Kameras rendern wieder, aber Overlay zeigt: Sim-Ansicht ≠ Datensatz-Ansicht (FOV, Kopf, Gierwinkel) — Abschnitt „Kopfkameras kalibrieren" |
| 17 | Kopfkamera-Kalibrierung (FOV 75°, Montagepunkt) bestätigt — Projektionsmodell trifft auf wenige Pixel — Abschnitt „Kopfkameras kalibrieren" |
| 18 | Stereobasis-Korrektur bestätigt — Kopfkameras vollständig kalibriert — Abschnitt „Kopfkameras kalibrieren" |
| 19 | Erste Domain-Gap-Neumessung nach Kalibrierung: Mittel 0,2831 — Gap sitzt in Überbelichtung, nicht Geometrie ([Ergebnis](#ergebnis-runs2026080819--der-gap-sitzt-in-der-belichtung-nicht-in-der-geometrie)) |
| 20 | Belichtungs-Sweep (2000→80) widerlegt Belichtung als Hebel gegen den Domain-Gap ([Sweep-Ergebnis](#sweep-ergebnis-runs2026080820--belichtung-ist-nicht-der-hebel)) |
| 21 | Albedo/Hintergrund als Ursache identifiziert (schwarze Hand, Boden-Textur) — halb gemessen ([Ergebnis](#ergebnis-runs2026080821--halb-gemessen-und-die-hälfte-war-schon-gelöst)) |
| 22 | Albedo-Fix bestätigt, Entscheidungskriterium (< 0,20) um 0,0056 verfehlt ([Ergebnis](#ergebnis-runs2026080822--albedo-bestätigt-kriterium-um-00056-verfehlt)) |
| 23 | Zwischenstand BC-Closed-Loop-Eval (2 von 20 Episoden): beide misslungen, volle Laufzeit ([Zwischenstand](#zwischenstand-runs2026080823-erste-2-von-20-episoden)) |
| 24 | Erste vollständige BC-Erfolgsraten-Messung: 0/20 — noch nicht aussagekräftig genug interpretiert ([Erste Messung](#erste-messung-runs2026080824--und-warum-sie-so-noch-nichts-entscheidet)) |
| 25 | Die eigentliche BC-Messung: Hand erreicht den Würfel, aber kein Erfolg ([Messung](#die-eigentliche-messung-runs2026080825--die-hand-ist-am-würfel)) |
| 26 | `grasp`-Test (Antwort): Griff scheitert auch ganz ohne Modell — `max_cube_lift_cm` 0,0 ([Antwort](#antwort-runs2026080826-der-griff-scheitert-auch-ohne-modell)) |
| 27 | Nachmessung: Finger schließen sichtbar — die Gelenkgrenze ist nicht die Ursache ([Nachmessung](#nachmessung-runs2026080827-die-finger-schließen--die-gelenkgrenze-ist-es-nicht)) |
| 28 | Messpunkt war zum dritten Mal falsch (distales Gelenk statt Fingerkuppe) ([Lauf 28](#lauf-28-der-messpunkt-war-zum-dritten-mal-falsch)) |
| 29 | Würfel hebt 7,9 cm ab (`GRASP_MODE=hold`, korrigierter Messpunkt) — Greif-Physik-Blocker gefallen, mit Vorbehalt ([Lauf 29](#lauf-29-der-würfel-hebt-ab-mit-einem-vorbehalt)) |
| 30 | Vorregistrierte Regel geschlossen: Politik kommandiert nur 19 % der Demo-Fingerspanne — greift nicht ([Lauf 30](#lauf-30-die-vorregistrierte-regel-ist-geschlossen--die-politik-greift-nicht)) |
| 31 | Meilenstein-Leiter im Einsatz: erste messbare Würfel-Anhebung im Closed Loop (0,22–1,03 cm) ([Lauf 31](#lauf-31-runs2026081203-die-leiter-im-einsatz--und-zwei-neue-zahlen)) |
| 32 | `span`-Gate entschieden: Median-Verhältnis 1,00 auf echten Bildern — Domain-Gap (a1) bestätigt, (a2) ausgeschlossen ([Lauf 32](#lauf-32-runs2026081301-das-span-gate-ist-entschieden--a1)) |
| 33/34 | `TUNE_VISUAL`-Checkpoint im Closed Loop: Fingerspanne 27,6 % (statt 20,5 %), `lifted` bleibt 0/10 ([Läufe 33/34](#läufe-3334-runs2026081403-runs2026081404-der-tune_visual-checkpoint-im-closed-loop)) |
| 35–37 | Würfellage-Rekonstruktion v4 zweimal abgelehnt; Kameramodell dagegen auf 0,3 cm bestätigt, Würfelebene lag 2 cm zu hoch ([Läufe 35–37](#läufe-3537-runs2026082201-03-die-würfellage-kommt-aus-dem-bild-nicht-aus-der-hand)) |
| 38–46 | Die vier Fingergrundgelenke standen in **jedem** Lauf still: der „Sign-Convention-Fix" drehte sie aus ihrer Gelenkgrenze, wo sie geklemmt wurden ([Läufe 38–46](#läufe-3846-runs2026082302-runs2026082405-12-die-fingergrundgelenke-standen-still)) |
| 47–51 | Drei unabhängige Geometriefehler ergaben zusammen ~12 cm: Vorzeichen, Beckenhöhe 0,85→0,764, Basis x 0→−0,057. Restabstand 4,5 cm ([Läufe 49–51](#läufe-4951-auch-die-basis-in-x--die-rekonstruktion-steht)) |
| 52 | Der **Gierwinkel** der Würfel wurde nie rekonstruiert — Sim stellte sie immer achsparallel. Schätzer gebaut und synthetisch abgenommen; auf echten Frames greifen 7 %, Ursache offen. Die Begründung stützte sich zunächst auf annotierte Debug-PNGs und ist zurückgezogen ([Lauf 52](#lauf-52-der-gierwinkel-fehlte-in-jedem-lauf)) |

---

### Treiber-Downgrade-Vorgehen (historisch, obsolet auf `ikr-ki-server-01`)

Notiert im Zuge der Segfault-Diagnose (siehe
[rl-anleitung.md, Troubleshooting](../weiterfuehrend/rl-anleitung.md#segfault-in-librtxscenedbpluginso--carbonpluginstartup-beim-start-rtx-pro-6000-blackwell)),
bevor feststand, dass der Treiber auf `ikr-ki-server-01` nicht änderbar ist. Als Referenz belassen,
falls der Server-Constraint sich ändert oder das Vorgehen für einen anderen Server relevant wird.

**Downgrade-Vorgehen (obsolet auf `ikr-ki-server-01` — Treiber dort fix, nicht änderbar; als Referenz
belassen, falls der Server-Constraint sich mal ändert oder für einen anderen Server relevant wird):**

1. **Vorher sichern, nichts löschen:**
   ```bash
   nvidia-smi --query-gpu=driver_version,name --format=csv   # aktuell: 610.43.02
   dpkg -l | grep nvidia-driver                                # exakter Paketname für Rollback
   uname -r                                                    # aktueller Kernel (7.0.0-28-generic)
   ```
2. **Verfügbarkeit von 580 prüfen, BEVOR etwas entfernt wird:**
   `apt-cache madison nvidia-driver-580-open` (gezielt die `-open`-Variante — neue Architekturen wie
   Blackwell werden zuverlässig nur über die offenen Kernel-Module unterstützt).
3. **Absicherung vor dem Wechsel:** Server-Konsole/IPMI-Zugriff sicherstellen (falls der Treiber nach
   dem Reboot nicht lädt, ist kein SSH über die GPU nötig, aber ein Fallback-Zugriffsweg schadet nicht).
   Kernel dabei **nicht** mitupdaten — nur den Treiber wechseln, um Variablen zu reduzieren.
4. **Wechsel:**
   ```bash
   sudo apt remove --purge 'nvidia-*'
   sudo apt install nvidia-driver-580-open
   sudo reboot
   ```
5. **Nach dem Reboot, in dieser Reihenfolge verifizieren:**
   - `nvidia-smi` lädt und zeigt `580.65.06` + die RTX PRO 6000 korrekt an?
   - **Regressionstest zuerst:** `./Simulation/server_robocasa_ref_run.sh preflight` — der bestehende,
     funktionierende RoboCasa-Workload darf durch den Treiberwechsel nicht kaputtgehen.
   - Erst danach: `./Simulation/server_rl_run.sh check` erneut.
6. **Falls 580 nicht bootet / GPU nicht erkannt wird:** zurück auf den in Schritt 1 notierten
   610er-Treiber (`sudo apt install nvidia-driver-610-open` o. ä.), RL-Pfad auf Pfad A (vast.ai)
   umstellen.


### Isaac-Sim-6.0-Migration (2026-08-07) — implementiert und seit 2026-08-08 auf Hardware bestätigt

[`Dockerfile.vastai`](../../Simulation/Dockerfile.vastai) wurde von `isaac-lab:2.3.2` auf
**`isaac-lab:3.0.0-beta2-post1`** (= Isaac Sim 6.0) portiert. Die Bundle-Angaben unten sind per
pip-list-Inventar aus dem echten Image verifiziert (2026-08-07), nicht aus Release Notes übernommen —
die nannten fälschlich torch 2.11. Folgeänderungen:

| Was | 2.3.2 (Isaac Sim 5.1) | 3.0.0-beta2-post1 (Isaac Sim 6.0) |
|---|---|---|
| Container-User am Ende des Basis-Image | `root` | **`isaaclab` (uid 1000)** → `USER root` im Dockerfile ergänzt, sonst scheitern alle `RUN`-Schritte an Rechten |
| Isaac-Sim-Python | 3.11 | **3.12** (Build-Guard schlägt fehl, falls das nicht stimmt) |
| Gebündeltes PyTorch | 2.7.0+cu128 | **2.10.0+cu128** (torchvision 0.25.0, numpy 2.5.1) |
| flash-attn-Wheel | `v2.7.4.post1 … cu12torch2.7 … cp311` | **`v2.8.1 … cu12torch2.10 … cp312`** — trifft alle Achsen exakt |
| gr00t-Laufzeit-Deps im Kit-Python | pandas kam aus dem 5.1-Bundle | 6.0-Bundle hat **kein pandas** mehr → `pandas==2.2.3` explizit gepinnt (erster Build brach mit `No module named 'pandas'` ab) |

Unverändert bleiben: Ubuntu 24.04 als Basis (deadsnakes-Python-3.10 für die GR00T-venv funktioniert
weiter), `${ISAACLAB_PATH}/_isaac_sim` als Symlink auf `/isaac-sim` (alle Pfad-Referenzen halten), und
die Isaac-Lab-Python-API (`isaaclab.sensors.TiledCamera`, `DirectRLEnv`, `ArticulationCfg` — die
Kamera-Deprecation in 6.0 betrifft `isaacsim.sensors.camera`, **nicht** Isaac Labs eigene Wrapper).
scipy/termcolor/wandb sind im 6.0-Bundle weiterhin enthalten und werden bewusst **nicht** gepinnt
(die uv.lock-Versionen wären Downgrades; `scipy==1.15.3` würde wegen `numpy<2.5` sogar das
Bundle-numpy mit-downgraden).

**⚠️ Verbleibendes Hauptrisiko — numpy 2.5:** gr00t ist gegen numpy 1.26.4 (uv.lock) entwickelt,
das 6.0-Bundle bringt **numpy 2.5.1** (nicht ersetzbar, Isaac Sim hängt daran). Import-Brüche fängt
der Smoke-Test im Dockerfile; ob das numerische Verhalten (Processor/Statistiken) identisch bleibt,
zeigt erst der erste echte Lauf. Dazu bleibt Isaac Lab 3.0.0-beta2 eine **Beta** — API-Abweichungen
in der Env-Konstruktion tauchen erst beim `check` auf.

**Ablauf für den ersten Test:**
```bash
./Simulation/update_sim_image.sh --vastai   # Rebuild auf Isaac Sim 6.0 (~60 min; Smoke-Test bricht bei Dep-Lücken ab)
# auf ikr-ki-server-01, nach git pull:
./Simulation/server_rl_run.sh clean          # alten Workbench-Container (5.1-Image) entsorgen!
./Simulation/server_rl_run.sh preflight      # Python 3.12 + torch 2.10 + flash-attn + gr00t auf der GPU
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh check   # der eigentliche Blackwell-Nachweis
```

### Würfel landen nicht auf dem Tisch (`num_envs > 1`) — GELÖST, zwei unabhängige Ursachen

**Symptom (2026-08-08):** Im RL-Lauf mit 4 Envs konnte `success_rate` gar nicht steigen. Der
Kamera-Dump zeigte `block_0` bei `z = 0.025` — die halbe Würfelkante, die Würfel lagen also **auf
dem Boden**.

**Ursache 1 — fehlender Env-Versatz.** `_reset_idx()` übergab die env-lokalen `block_*_range`-Werte
direkt an `write_root_pose_to_sim()`, das **Weltkoordinaten** erwartet; `self.scene.env_origins`
fehlte. Die Würfel landeten damit am Weltursprung, wo bei `num_envs > 1` kein Tisch steht (jede Env
hat ihren eigenen bei `env_origin + (0.5, 0, …)`), und fielen durch.
**Mit `num_envs=1` ist der Fehler unsichtbar** (Env-Ursprung `(0,0,0)`) — deshalb fiel er in allen
Sim-Evals und Replay-Läufen nie auf und erst im RL-Lauf mit mehreren Envs.
Behoben mit `block_pos += self.scene.env_origins[env_ids]`; in `runs/20260808/09` nachgemessen:
Würfel bei Welt `(1.37, -1.17, 0.895)`, also auf dem Tisch von env 0 (Oberkante 0.87 + 0.025).

**Ursache 2 — überlappende Startpositionen.** Alle drei Würfel wurden unabhängig aus **demselben**
Rechteck gezogen (x 0.30–0.40, y −0.20–0.20). Zwei 5-cm-Würfel überlappen dort mit ~18 % je Paar,
bei drei Paaren also in **44,9 %** aller Resets (Monte-Carlo, 200 000 Ziehungen). PhysX löst die
Durchdringung auf, indem es die Würfel auseinanderschießt — in `runs/20260808/08` lagen sie danach
gut 1 m entfernt am Boden. Behoben mit disjunkten y-Bändern je Würfel plus 3 cm Rand:
Überlappungsrate 0,00 %, garantierter Mindestabstand 6 cm bei 5 cm Kantenlänge, x bleibt voll
randomisiert und die y-Gesamtspanne unverändert.

**Nicht betroffen:** Reward und Erfolgskriterium rechnen ausschließlich mit *Differenzen*
(`cdist` Hand↔Würfel, paarweise Würfelabstände, Höhendifferenz) und sind damit frame-unabhängig.
Ebenso die per `init_state` gespawnten Objekte — Tisch und Roboter sind nachgemessen korrekt
(`table` Welt `(1.5, -1, 0.435)`, `robot_root` `(1, -1, 0.85)` bei `env_origin (1, -1, 0)`), Isaac
Lab setzt die selbst env-relativ. Und die Juni-Auswertungen liefen mit `num_envs=1`, sind also
unberührt; der 0-%-Befund dort bleibt der Domain-Gap.

**Folge:** Alle RL-Läufe vor dem Fix mit `RL_NUM_ENVS` 2 oder 4 trainierten auf einer unlösbaren
Aufgabe — die Würfel lagen außerhalb der Reichweite am Boden. `reward_mean` bewegte sich trotzdem,
weil der Shaped Reward aus Gelenk- und Abstandsgrößen kommt.

### Kamerabilder gleichmäßig weiß — GELÖST (`runs/20260808/13` + `14`)

**Kurzfassung für Eilige:** Die Kamera-Prims standen in der USD-Stage **anders ausgerichtet** als
`cam.data` meldete — 95,6° bei den High-Cams, 102,8° an den Handgelenken, 136,8° bei `cam_scene`.
Die Kameras filmten den Himmel. `_setup_scene()` schreibt die Orientierung jetzt selbst
(`RL_CAM_USD_FIX=1`, Default an). In Lauf 14 zeigen `cam_left_high`, `cam_right_high` und
`cam_scene` wieder die Szene; die beiden Wrist-Cams wurden dort wegen eines Namensfehlers noch
übersprungen und sind seitdem nachgezogen. Herleitung unten.

**Stand 2026-08-08 nach `runs/20260808/09`:** Der Würfel-Fix oben ist wirksam, die Bilder sind
trotzdem gleichmäßig hell (min/median/max **245/248/249**, chroma 4,0, dunkel 0 %). Die Szene ist
damit als Ursache ausgeschlossen — und zwar gemessen, nicht vermutet:

| geprüft | Ergebnis |
|---|---|
| Objektposen zur Laufzeit | Tisch, Roboter, Würfel alle korrekt in ihrer Env |
| Würfel im Blickfeld? | `IM BILD` bei 0,66–0,71 m, u/v deutlich innerhalb des Frustums |
| Kamerapose + Konvention | alle fünf Kameras `quat_w_world / +X` bei **0,0°** Abweichung — ⚠️ **diese Zeile war die Falle**, siehe Lauf 13 |
| Clipping-Ranges | Objekte liegen in allen Kameras innerhalb |

**Belichtung: widerlegt** (`runs/20260808/10`, `RL_DOME_SWEEP=500,120,30,8`). Das Bild wird nur
dunkler, es kommt keine Struktur zum Vorschein:

| intensity | 2000 | 500 | 120 | 30 | 8 |
|---|---|---|---|---|---|
| min/median/max | 227/233/234 | 182/193/201 | 83/96/97 | 25/31/32 | 5/7/7 |
| Spannweite | 7 | 19 | 14 | 7 | 2 |

**Was tatsächlich gerendert wird** (Kantenenergie und dominanter Farbanteil je PNG):

| Kamera | Kanten x/y | eine Farbe | Befund |
|---|---|---|---|
| `cam_left_high` | 0.00 / 0.00 | 100,0 % | konstanter Puffer, **nichts** gerendert |
| `cam_right_high` | 0.00 / 0.00 | 100,0 % | konstanter Puffer |
| `cam_right_wrist` | 0.00 / 0.00 | 100,0 % | konstanter Puffer |
| `cam_left_wrist` | 0.29 / 0.14 | 91,0 % | **DEX3-Hand klar sichtbar**, Rest Hintergrund |
| `cam_scene` | 0.34 / 0.47 | 93,7 % | Bodengitter im Eck, Rest Hintergrund |

~~Ein fehlgerichtetes Objektiv in einer beleuchteten Szene zeigt *irgendetwas*. Ein über alle fünf
Belichtungen exakt einfarbiges Bild ist kein Blickwinkel-, sondern ein Renderproblem.~~
**Falsch, und dieser Fehlschluss hat drei Läufe gekostet.** Eine Kamera, die in den leeren Himmel
über einer Domelight-Kuppel blickt, zeigt eben *nicht* irgendetwas, sondern exakt eine Farbe. Der
Schluss „einfarbig ⇒ kein Blickwinkelproblem" war genau verkehrt herum (Lauf 13).

**Domain Randomization: widerlegt** (`runs/20260808/11`, `DR_ENABLED=0`). Die DR lieferte nur die
Chroma (5,76 → 0,00), nicht die Objekte. Schärfer noch, gemessen an der Zahl verschiedener
Grauwerte im ganzen 640×480-Bild:

| Kamera | Grauwerte | Befund |
|---|---|---|
| `cam_left_high` | **1** | nur Dome-Farbe (Lauf 12: kanalweise nachgewiesen) |
| `cam_right_high` | **1** | nur Dome-Farbe |
| `cam_right_wrist` | **1** | nur Dome-Farbe |
| `cam_left_wrist` | 210 | Hand sichtbar |
| `cam_scene` | 156 | Boden sichtbar |

Ein einziger Grauwert ist kein „zu wenig Kontrast", das ist ein Sensor, der nichts von der Szene
schreibt. (Damals als „unbeschriebener Puffer" gelesen — Lauf 12 zeigt, dass es die Dome-Farbe
ist, also ein korrekt gerendertes Bild von leerem Raum. Das verschiebt die Ursache vom Sensor auf
das, was der Renderer an dieser Stelle vorfindet.)
`cam_left_high` blickt *laut `cam.data`* mit −47,7° nach unten; zwischen x ≈ 0,9 m und 2,56 m
müsste derselbe globale Boden im Bild stehen, den `cam_scene` zeigt, dazu die eigenen Arme wie im
Dataset-Referenzbild. Nichts davon. Und `cam_left_wrist` und `cam_right_wrist` unterscheiden sich
in nichts außer dem Link, an dem sie hängen — trotzdem zeigt nur eine von beiden etwas.
~~Damit sind sowohl „Würfel rendern nicht" als auch jede Pose-Erklärung raus.~~ **Auch das war
verkehrt:** dass ausgerechnet eine der beiden baugleichen Wrist-Cams etwas zeigt, ist der
*stärkste* Hinweis auf ein Pose-Problem — die eine blickt in den Roboter, die andere von ihm weg
(Lauf 13).

**Sensorklasse: widerlegt** (`runs/20260808/12`, `RL_CAMERA_CLASS=camera`). Alle fünf Sensoren
liefen als gewöhnliche `Camera` statt `TiledCamera`, bei identischer Pose, Optik und Auflösung —
das Ergebnis ist dasselbe: `cam_left_high` und `cam_right_high` haben 2 Grauwerte (Spannweite 1,4
bei std 0,20 — Denoiser-Rauschen), `cam_right_wrist` genau 1. `TiledCamera` ist damit raus, der
Fehler sitzt in Szene oder Render-Setup.

Lauf 12 hat aber die Ursache eingekreist, weil die Bilder erstmals der Diagnostik **widersprechen**:

| | `cam_left_high` | `cam_scene` |
|---|---|---|
| Diagnostik sagt | Tisch `u=+0.02 v=−0.87`, 1,13 m, **IM BILD** | Tisch, Roboter, alle 3 Würfel **IM BILD**, 2,2–2,55 m |
| Bild zeigt | reine Dome-Farbe | Bodengitter im Eck, sonst reine Dome-Farbe |

Die „leeren" Frames sind kanalweise **exakt** die Dome-Farbe (R 246 / G 244 / B 241 bei
Intensität 2000, R 93 / G 85 / B 76 bei 120) — der 0,75-Graupunkt des `DomeLightCfg` durchs
Tonemapping. Das ist kein unbeschriebener Puffer, das ist ein korrekt gerendertes Bild **von
nichts**: der Renderer sieht an dieser Stelle Himmel. Und `cam_scene` zeigt es am deutlichsten —
Boden ja, Tisch und Roboter mitten im Bild nein. Rechnerisch müsste bei Pitch −22,3° und 47,2°
vertikalem Öffnungswinkel der Boden 97 % des Frames füllen; sichtbar ist er auf ~8 %.

Gleichzeitig **rendert** `cam_left_wrist` die DEX3-Hand sauber, inklusive UNITREE-Schriftzug — das
Roboter-USD ist also im Render-Graph. Nur eben nicht dort, wo `cam.data` es verortet.

**Auflösung: die USD-Stage als unabhängiger Zeuge** (`runs/20260808/13`). Bisher stammten
Soll-Richtung, Treffer-Matrix und Frustum-Rechnung alle aus `data.quat_w_world`; ein Widerspruch
zwischen zwei Ableitungen aus einer Quelle ist mit dieser Quelle nicht auflösbar.
`dump_camera_poses.py` liest deshalb zusätzlich direkt aus der Stage — und der erste Lauf
entscheidet die Sache:

| Kamera | Prim-Position | Blick laut Stage | Blick laut `cam.data` | Abweichung |
|---|---|---|---|---|
| `cam_left_high` | `[1.0, −0.92, 1.45]` | `[0.673, 0, 0.739]` | `[0.666, −0.097, −0.739]` | **95,6°** |
| `cam_right_high` | `[1.0, −1.08, 1.45]` | `[0.673, 0, 0.739]` | `[0.666, +0.097, −0.739]` | **95,6°** |
| `cam_left_wrist` | `[1.16, −0.851, 1.045]` | `[0, −0.882, 0.471]` | `[0.882, 0, −0.471]` | **102,8°** |
| `cam_right_wrist` | `[1.16, −1.149, 1.045]` | `[0, −0.882, 0.471]` | `[0.882, 0, −0.471]` | **102,8°** |
| `cam_scene` | `[2.8, 0.6, 1.7]` | `[0.925, 0, 0.38]` | `[−0.633, −0.675, −0.38]` | **136,8°** |

**Position und Optik stimmen exakt** — auch `focalLength`/`aperture` am Prim ergeben genau die
47,2° × 36,3°, die der Renderer meldet. Es ist ausschließlich die Rotation.

*Warum `cam.data` das nicht sehen kann:* Isaac Lab rechnet die Offset-Rotation beim Spawn von
`convention="world"` nach OpenGL um und beim Lesen wieder zurück. Ist diese Umrechnung fehlerhaft,
hebt der Rückweg sie auf — `quat_w_world` liefert brav die Config zurück, während der Renderer die
verdrehte Pose benutzt. Die Treffer-Matrix konnte diesen Fehler also **prinzipiell** nicht finden.

*Warum die Stage-Pose die gerenderte ist:* weil sie alle fünf Bilder erklärt, Zug um Zug, während
die gemeldete Pose keines erklärt.

| Kamera | Stage-Pose zeigt auf | Bild |
|---|---|---|
| `cam_left_high` / `cam_right_high` | +42,6° nach **oben** — Tisch liegt darunter | reine Dome-Farbe ✔ |
| `cam_right_wrist` | −Y, vom Roboter **weg**, +28° nach oben | genau **1** Grauwert ✔ |
| `cam_left_wrist` | −Y, in den Roboter **hinein**, 0,15 m | DEX3-Hand formatfüllend ✔ |
| `cam_scene` | +X und +22° nach oben, Frame reicht bis −1,3° unter den Horizont | schmaler Bodenkeil unten ✔ |

Besonders `cam_right_wrist` ist beweiskräftig: unter der *gemeldeten* Pose stünde `block_0` in
0,22 m Entfernung mitten im Bild — ein Würfel auf 22 cm rendert garantiert. Unter der Stage-Pose
ist dort leerer Raum. Das Bild hat genau einen Grauwert.

Bei den drei statischen Kameras hat der Fehler eine saubere Signatur: die gerenderte Blickrichtung
ist `(√(fx²+fy²), 0, −fz)` der gemeldeten — **der Yaw wird verworfen und das Vorzeichen des Pitch
gekippt**. Für die Handgelenke fällt das anders aus, weil dort die Link-Rotation dazwischenliegt.
Als feste Konventionsmatrix oder als Vertauschung der Quaternion-Komponenten ließ sich das nicht
nachbauen; der genaue Mechanismus in Isaac Lab 3.0-beta2 bleibt offen — für die Reparatur ist er
auch nicht nötig.

**Der Fix — die Orientierung selbst schreiben.** `_setup_scene()` ruft am Ende
`_force_camera_prim_orientations()`: für jede Kamera wird aus dem Config-Quaternion (Welt-Konvention)
die USD-Orientierung gebildet — `R_usd = R_welt · C` mit den Spalten von `C` als USD-Achsen
(`X_usd = −Y_welt`, `Y_usd = +Z_welt`, `Z_usd = −X_welt`) — und als `xformOp:orient` direkt auf das
Prim geschrieben. Der Op-Stack wird dabei neu aufgebaut (Translate + Orient), damit kein
konkurrierender `rotateXYZ` danebensteht. Die Umrechnung ist offline gegen alle vier Config-Posen
geprüft: Rückweg über die Quaternion trifft die Soll-Blickachse auf < 0,03°.

Abschalten mit `RL_CAM_USD_FIX=0` — dann bleibt die Isaac-Lab-Variante stehen, für den direkten
Vergleich im selben Dump.

Zusätzlich behoben: `ComputePurpose()` nimmt in USD 25.11 keine `TimeCode` mehr, der Geometrieblock
des Dumps ist daran abgestürzt (Zeile 212). Deshalb fehlen `visibility`/`purpose`/BBox in Lauf 13
noch; beide Signaturen werden jetzt bedient.

```bash
git pull
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh cams
```

**Bestätigt in `runs/20260808/14`** — und zwar an den Bildern gemessen, nicht an der Diagnostik:

| Kamera | min/median/max | chroma | dunkel % | Bild |
|---|---|---|---|---|
| `cam_left_high` | 0 / 229 / 241 | 6,93 | 26,9 % | Tisch, Würfel, eigene Hand |
| `cam_right_high` | 0 / 232 / 242 | 7,83 | 18,7 % | dito, gespiegelt |
| `cam_scene` | 10 / 129 / 243 | 7,75 | 32,1 % | ganze Szene: Roboter, Tisch, drei Würfel |
| Juni-Referenz (Isaac Sim 4.x) | 32 / 229 / 239 | 6,0 | 11–22 % | — |

Die High-Cams treffen die Referenz von vor der Migration praktisch exakt. Der Geometrieblock lief
diesmal durch und ist unauffällig: Boden, Tisch, alle drei Würfel, Band und Roboter stehen
`inherited` / `purpose=default` mit korrekten Welt-BBoxen in der Stage — die Hypothesen
„ausgeblendet" und „keine renderbare Geometrie" sind damit beide erledigt.

**Zwei Nachträge aus Lauf 14:**

1. `cam.data` meldet seit dem Fix eine *falsche* Blickrichtung (`cam_left_high`: 174,4° neben dem
   Prim), während das Prim exakt auf der SOLL-Richtung sitzt — dieselbe kaputte Umrechnung, nur
   rückwärts, und damit ein zweiter unabhängiger Beleg. Praktische Folge: **Frustum-Rechnung und
   Treffer-Matrix im Dump sind jetzt unbrauchbar**, Referenz ist das Bild. Der Stage-Abgleich prüft
   deshalb neu gegen die *lokale* Config-Rotation im selben Elternframe (Spalte `SOLL:`); die
   `cam.data`-Spalte bleibt nur noch als Anzeige der Isaac-Lab-Umrechnung stehen.
2. Die beiden Wrist-Cams wurden übersprungen — sie heißen in `CAMERA_CFG`
   `cam_left_wrist_local` / `cam_right_wrist_local`, der Fix suchte unter dem Sensornamen. Er liest
   die Offset-Rotation jetzt aus `cam.cfg.offset`, also aus genau der Quelle, aus der auch Isaac Lab
   beim Spawn liest, und deckt damit alle fünf Kameras unabhängig von der Benennung ab.

**Abgeschlossen mit `runs/20260808/15`.** 20 Prims (5 Kameras × 4 Envs) neu ausgerichtet,
`SOLL: OK (0.0°)` bei allen fünf, und die Bilder bestätigen es:

| Kamera | Graustufen | Kantenenergie | Bild |
|---|---|---|---|
| `cam_left_high` | 248 | 2,12 | Tisch, Würfel, beide Hände |
| `cam_right_high` | 240 | 2,11 | dito, dazu das `stack_band` |
| `cam_left_wrist` | 156 | 1,25 | Unterarm + Hand, gelber und grüner Würfel |
| `cam_right_wrist` | 206 | 1,10 | Unterarm + Hand, Tischkante |
| `cam_scene` | 223 | 5,06 | ganze Szene |

Zum Vergleich Lauf 12 (leer): 1–3 Graustufen bei Kantenenergie 0,00–0,31. Die Kameras sind damit
repariert.

**Die Selbstbewertung des Dumps musste dafür ausgetauscht werden.** Sie meldete in Lauf 15 drei
intakte Kameras als „kein Kontrast, Bild praktisch leer", weil ihr Kriterium der Anteil dunkler
Pixel war (`dunkel < 5 %`). Ein weißer Roboterarm vor einer weißen Tischplatte hat 0,0 % dunkle
Pixel und ist trotzdem vollständig korrekt. Der Test läuft jetzt über **Graustufenzahl und
Kantenenergie** — die messen, *ob* etwas gerendert wurde, statt *wie hell* es ist. An den echten
Daten trennt das sauber: leere Frames 0,00–0,44, intakte 1,10–5,06. `dunkel%` und `chroma` bleiben
als beschreibende Spalten stehen, ohne Urteil.

Offen bleibt eine **Kalibrierfrage, kein Renderfehler**: `cam_right_wrist` blickt an der Tischkante
vorbei ins Bodengitter, während `cam_left_wrist` die Würfel im Bild hat. Das deckt sich mit der
schon in `g1_dex3_cfg.py` notierten Asymmetrie (Iteration 9: „~45°-Diagonal-Versatz rechts = reale
Pose-Differenz"). Die konnte bisher niemand nachjustieren, weil die Kamera gar nichts gerendert
hat — jetzt geht es.

> **Konsequenz für alle bisherigen Ergebnisse:** Jeder Sim-Lauf seit der Isaac-Sim-6.0-Migration
> hat die Policy auf Himmelsbildern laufen lassen. Erfolgsraten, Reward-Kurven und
> Domain-Gap-Zahlen aus dieser Zeit sind gegenstandslos und müssen neu erhoben werden.

**Bestätigt nebenbei** (beides in Lauf 11 im Log): Die Renderer-Intrinsik meldet 47,2° × 36,3° für
die High-Cams — genau das, was die vorher angenommene `horizontal_aperture` ergab, die
Frustum-Rechnung war also korrekt. (Dass diese 47,2° *selbst* falsch waren — projektiert sind 75° —
fiel erst bei der Kalibrierung auf, siehe Iteration 13 weiter unten.) Und der Roll ist 0,0° bei beiden High-Cams und `cam_scene`
(−90° an den Handgelenken, dort armposenabhängig) — beides allerdings wieder aus `cam.data` und
damit nach Lauf 13 hinfällig, solange der Fix nicht bestätigt ist. Die Diagonale in `cam_scene`
hielt ich für die Kante der endlichen Bodenplatte; sie ist der streifende Blick knapp über den
Horizont — sichtbar sind nur ~8 % Boden statt der rechnerischen 97 %.

> **Nebenbefund:** `_randomize_visuals()` würfelt die Dome-Intensität pro Episode neu
> (`uniform(1000, 3800)`). `RL_DOME_INTENSITY` wirkt daher nur bei `DR_ENABLED=0` dauerhaft.

**Drei Lehren aus dieser Fehlersuche:**

1. Der Dump druckte die Objektposen env-relativ unter der Überschrift „WELTPOSITION". Das hat zu
   einer kompletten Fehldiagnose geführt („alle Objekte am Weltursprung"), obwohl die Szene
   korrekt war. Er gibt jetzt **beide** Spalten aus.
2. Die Treffer-Matrix („`quat_w_world/+X` bei 0,0°") vergleicht die *gerenderte* Quaternion gegen
   eine Soll-Richtung aus **derselben** Config mit **derselben** Hilfsfunktion. Sie beweist, dass
   Isaac Lab umsetzt, was verlangt wird — nicht, dass das Verlangte stimmt. Und sie prüft nur die
   *Blickachse*, nicht die Drehung um sie. Der Dump gibt deshalb jetzt zusätzlich den **Roll**
   gegen Welt-Oben aus und stützt die Frustum-Rechnung auf `data.intrinsic_matrices` statt auf
   eine angenommene `horizontal_aperture`.
3. Und das ist die eigentliche Lehre: **eine Größe, die durch einen Hin- und Rückweg derselben
   Umrechnung läuft, kann diese Umrechnung nicht prüfen.** `quat_w_world` meldete fünf Läufe lang
   0,0° Abweichung, während der Renderer 96–137° danebenlag. Vier Hypothesen (Belichtung, DR,
   Anti-Aliasing, Sensorklasse) wurden widerlegt, bevor jemand die Stage selbst gefragt hat. Wenn
   Diagnostik und Beobachtung sich widersprechen, ist die nächste Messung nicht die fünfte
   Variante der Diagnostik, sondern **eine zweite Quelle**.

### Historisch: DLSS-Upscaling (nicht die Ursache)

**Symptom (2026-08-08, RTX PRO 6000 Blackwell):** Alle fünf Kameras liefern nahezu einfarbige
Bilder. `cam_left_high` spannte über das gesamte Bild nur die Helligkeitsstufen **244–249**; die
Chroma fiel von 6,0 (Juni, mit farbigen Würfeln) auf 2,0. Die Wrist-Kameras erwischten noch einen
Streifen Hand am Bildrand, wo im Juni die Hand formatfüllend war. Zum Vergleich der Juni-Frame:
`Simulation/old_videos/12/_debug_obs_cam_left_high.png` (Tisch, drei Würfel, beide Hände).

**Was es NICHT ist** — drei per Messung ausgeschlossene Verdächtige:

| Verdacht | Widerlegt durch |
|---|---|
| Kamera-Pose / -Orientierung falsch | `server_rl_run.sh cams`: für alle fünf Kameras Position deckungsgleich mit dem Offset, Blickrichtung deckungsgleich mit der Config, `quat_w_world/+X` bei **0,0°** |
| Falsche Konvention (`convention="world"` als ROS/OpenGL interpretiert) | Treffer-Matrix im Dump: `quat_w_world/+X` gewinnt bei 0,0°; ROS und OpenGL liegen 22–130° daneben |
| `TiledCamera`-Slicing bei mehreren Envs | Lauf mit `RL_NUM_ENVS=1` zeigt dasselbe Bild wie mit 4 |
| Überbelichtung / Sättigung | Kein Kanal erreicht 255 (Max 247–249), und `cam_left_wrist` hat min=10 — Kontrast ist vorhanden |
| DLSS-Upscaling unter Mindestauflösung | `RL_AA_MODE=DLAA` beseitigte die Warnung, verschlechterte das Bild aber (Chroma 5,93 → 0,96); `RL_AA_MODE=Off` ändert nichts |
| Render-Konvergenz (zu wenige Frames) | `RL_SETTLE_STEPS=60` macht die Läufe deterministisch (Median stabil 232) — aber weiterhin leer |
| Clipping-Range zu eng | High-Kameras: `(0.1, 20.0)`, Tisch bei 0,79 m; Wrist: `(0.01, 5.0)`; Szene: `(0.1, 30.0)` — alles bequem drin |

**Was es ist (Kit-Log):**

```
[Warning] [omni.rtx] DLSS increasing input dimensions:
    Render resolution of (320, 240) is below minimal input resolution of 300.
```

DLSS rendert intern auf halber Auflösung (640×480 → 320×240) und liegt damit unter seinem
eigenen Minimum. Die Env setzte bis dahin **keine** Render-Konfiguration, lief also auf den
Isaac-Sim-6.0-Defaults.

**Auch das war es nicht.** `RenderCfg(antialiasing_mode="DLAA")` ließ sich sauber setzen (alle
Felder akzeptiert, die DLSS-Warnung verschwand), machte die Bilder aber **schlechter**: Chroma
5,93 → 0,96, Wertebereich auf 246–248 geschrumpft. DLAA ist selbst ein temporales Verfahren.
Der Modus ist deshalb **kein Default mehr**, sondern ein Messhebel (`RL_AA_MODE`).

**Die zwei belastbarsten Spuren** — beide aus den Messreihen, nicht aus Logzeilen:

1. **Die dunklen Bildbereiche fehlen.** Im Juni waren 11–22 % der Pixel dunkler als Helligkeit
   100 (Hintergrund und Schatten), heute 0 %. Ein weißer Tisch vor weißem Hintergrund ist
   unsichtbar, egal wie exakt die Kamera steht. Das deutet auf Beleuchtung/Hintergrund.
2. **Die Läufe schwanken.** Bei identischer Szene ergaben aufeinanderfolgende Läufe deutlich
   verschiedene Helligkeiten und Chroma-Werte. Ein konvergierter Render ist deterministisch —
   das deutet auf zu wenige Render-Frames vor der Messung (temporaler Denoiser).

**Drei Hebel zum Eingrenzen**, je ein `cams`-Lauf (~2 min):

```bash
# 1) Render-Konvergenz: viel länger settlen lassen
#    ERLEDIGT: settle=60 macht die Läufe reproduzierbar (median stabil 232), bleibt aber weiß.
RL_SETTLE_STEPS=60 HF_TOKEN=hf_... ./Simulation/server_rl_run.sh cams
# 2) Anti-Aliasing-Modus durchprobieren (Off|FXAA|DLAA|DLSS|TAA; leer = Isaac-Default)
#    ERLEDIGT für DLAA: griff sauber, machte die Bilder aber SCHLECHTER (chroma 5,93 -> 0,96).
RL_AA_MODE=Off RL_SETTLE_STEPS=60 HF_TOKEN=hf_... ./Simulation/server_rl_run.sh cams
# 3) Belichtung: mehrere Dome-Intensitäten in EINEM Lauf (aktuell der beste Verdacht)
RL_DOME_SWEEP=500,120,30,8 HF_TOKEN=hf_... ./Simulation/server_rl_run.sh cams
```

`cams` gibt selbst eine Kennzahlen-Tabelle aus (min/median/max, Graustufen, Kantenenergie, chroma,
dunkel-%) samt der Juni-Referenzwerte und markiert strukturlose Bilder mit
`<-- keine Struktur, nichts gerendert`. Damit ist der Vergleich direkt im Log ablesbar, ohne die
PNGs auszuwerten.

> **Folge fürs Training:** Solange die Policy-Kameras leer sind, sieht das Modell nichts — RL
> optimiert dann gegen ein blindes Modell, während `reward_mean` sich weiter bewegt (der Shaped
> Reward kommt aus Gelenkpositionen). Erst `cams` grün, dann RL starten. Die Juni-Auswertungen
> sind unberührt: die liefen auf korrekt gerenderten Kameras.

### Kopfkameras kalibrieren — Iteration 13 (2026-08-08), Renderprüfung offen

Nachdem die Kameras wieder rendern (`runs/20260808/16`), zeigt der Overlay gegen
`Simulation/camera_reference/`, dass die Sim-Ansicht **nicht** die Ansicht des Datensatzes ist.
Drei Abweichungen, alle gegen eine unabhängige Quelle geprüft statt geschätzt:

| Befund | Beleg | Korrektur |
|---|---|---|
| **Sichtfeld 47,2° statt 75°** | `hfov_deg = 75.0` stand seit jeher in `g1_dex3_cfg.py`, wurde aber **nirgends gelesen** — gespawnt wurde die fest eingetragene `focal_length=24.0` | `focal_high/_wrist/_scene` aus `hfov_*_deg` berechnet, Env liest sie. 75° → 13,65 mm |
| **Kopf verdeckt ein Bilddrittel** | Kameras auf `z=1.45`, `y=±0.08` — neben und über dem Kopf. Der echte G1 trägt sie laut URDF im `d435_link`, pelvis-relativ `(0.0537, 0.0175, 0.4739)` | Montagepunkt env-lokal `(0.0537, 0.0175, 1.3239)`; Nahebene 0,1 → 0,15 m als Absicherung gegen die Kopfschale |
| **Schräge Tischkante** | Beide Kameras zielten auf **einen** Punkt → ±8,3° Gierwinkel. Ein reales Stereopaar blickt parallel | Ziel je Kamera auf der eigenen y-Linie, Gier exakt 0,00° |

Die **Stereobasis** kam aus den Referenzbildern selbst: Querversatz 40 px (SAD-Minimum über die
Tischplatte), Würfelkante 40–46 px bei bekannten 5 cm → Motivabstand ~0,57 m → Basis ~4,7 cm. Der
Wert hängt nicht am angenommenen FOV, weil Abstand und Winkel gemeinsam mitskalieren; er deckt sich
mit den 50 mm einer RealSense D435. Statt ±0,08 (16 cm) also **±0,025**.

Vorausberechnung der neuen Bildaufteilung (Blickachse −55,0°, Gier 0,00°, 75° × 59,8°):

| Punkt | Bildzeile von 480 | |
|---|---|---|
| Tisch-Hinterkante | 29 | Tisch vollständig im Bild wie in der Referenz |
| Würfel (Mitte) | 240 | exakt Bildmitte |
| rechte / linke Hand | 322 / 355 | von unten ins Bild, wie in der Referenz |
| Tisch-Vorderkante | 473 | gerade noch drin |

**Die Handgelenkskameras bleiben unangetastet** — bewusst. Der Stage-Abgleich zeigt für beide
dieselbe lokale Blickrichtung (`[0.882, 0, −0.471]`), sie sind also identisch konfiguriert. Dass
`cam_right_wrist` die Würfel verfehlt und `cam_left_wrist` nicht, kommt somit von der Armpose, nicht
von der Kamera. Eine einseitige Korrektur würde die Symmetrie zerstören, die der reale Roboter hat.

Nächster Schritt: rendern und den Overlay erneut ansetzen.

```bash
git pull
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh cams
# Frames herunterladen, dann lokal:
python Simulation/scripts/overlay_camera_check.py --sim-dir Simulation/runs/<datum>/<nr>/cam_dump
```

Im Overlay muss der Tisch beide Bilder füllen, die Hinterkante annähernd waagerecht liegen und der
eigene Kopf verschwunden sein. Taucht der Kopf weiter auf, sitzt der `d435`-Punkt innerhalb der
Kopfschale — dann `left_high_eye`/`right_high_eye` um 2–3 cm in +X schieben.

**Ergebnis in `runs/20260808/17`:** Der Kopf ist weg, der Tisch füllt symmetrisch das Bild, beide
Hände kommen von unten herein, das Log meldet 75,0° × 59,8°. Zwei Messungen aus dem Overlay:

* **Sichtfeld bestätigt.** Die 5-cm-Würfel messen real 45 × 50 px und in der Sim 45 × 53 px — bei
  vergleichbarem Motivabstand (0,52–0,54 m gegenüber ~0,57 m). Mit den alten 47,2° wären sie rund
  1,6-fach zu groß gewesen.
* **Das Projektionsmodell trägt.** Rechnet man die im Log protokollierten Würfelpositionen durch die
  neue Kamera, landen sie im gerenderten Bild auf ±wenigen Pixeln (grün exakt, mittlere Abweichung
  −6/+6 px; die Ausreißer sind der von der Hand halb verdeckte rote Würfel und der
  Schwerpunkt-Bias der sichtbaren Würfelfläche). Framing lässt sich damit **vorausrechnen**, statt
  es zu errendern — jede weitere Iteration kostet keinen Renderlauf mehr.

**Iteration 14 (aus Lauf 17):** Die Stereobasis wird um `y=0` zentriert statt um die `y=0.0175` des
`d435_link`. In der Reset-Pose stehen die Handgelenke fast symmetrisch (`y=+0.158` / `−0.144`), und
im Referenzbild liegen beide Hände symmetrisch um die Bildmitte — die reale Kamera sitzt also auf
der Mittellinie. Das `d435_link` ist der Montageflansch eines Moduls, nicht der Mittelpunkt
zwischen zwei Bildsensoren. `x` und `z` bleiben beim URDF-Wert, die sind eindeutig. Rechnerisch
rückt die Hand-Mitte damit von 359 auf 341 px (rechte Kamera spiegelbildlich 307 → 289), das Paar
liegt also symmetrisch um die Bildmitte statt 13 px daneben.

**Bestätigt in `runs/20260808/18`.** Die Stereobasis ist damit nicht mehr behauptet, sondern
gemessen — an drei unabhängigen Stellen:

| Prüfung | Ergebnis |
|---|---|
| Vorhersage vs. Rendering, grüner Würfel | links Δ = (−0, −0) px, rechts Δ = (−2, −0) px |
| Disparität links/rechts, gerendert | +44,1 px (rot) / +43,9 px (grün) |
| Disparität aus dem Modell | +42,1 px — **Referenzbilder real: +40 px** |

Damit stimmen Sichtfeld, Montagepunkt und Basis. Der rote Würfel liegt in beiden Kameras um dieselben
−17/−20 px daneben: das ist der Schwerpunkt-Bias der halb von der Hand verdeckten Fläche, kein
Kameraversatz — ein Kamerafehler wäre nicht in beiden Bildern identisch.

**Damit sind die Kopfkameras kalibriert.** Was im Overlay noch verschieden aussieht, sind keine
Kameraparameter:

* **Armpose.** Das Referenzbild zeigt die Arme mitten in der Aufgabe (Hände erhoben, Finger nach
  oben), die Sim steht in der Reset-Pose. Handpositionen sind zwischen beiden nicht vergleichbar.
* **Würfelplatzierung.** Die Sim würfelt sie pro Episode neu. In Lauf 17 lag ihr Schwerpunkt 28 px
  über dem der Referenz, in Lauf 18 — bei identischer Kamera — 6 px darunter. Der Versatz misst den
  Zufallsgenerator, nicht die Pose.
* **Abstand zum Tisch.** Der Tisch überspannt in der Sim vertikal 52,1°, im Referenzbild nur 42,3°.
  Bei gleicher Tischtiefe entspricht das einem Roboter, der **~15 cm weiter vom Tisch weg** sitzt.
  Das ist Szenenlayout, und es zu ändern verschiebt den erreichbaren Greifraum, der aus den
  Replay-Daten abgeleitet wurde (`block_x_range`, Kommentar in `g1_dex3_blockstack_env.py`).
  Notiert als Entscheidung, nicht als Fix.

Nächster Schritt ist damit nicht die nächste Overlay-Runde, sondern **Schritt 2: den Domain-Gap neu
messen** (`Simulation/scripts/measure_domain_gap.py`). Der liefert eine Zahl statt eines
Augenmaßes; die letzten Werte (Mittel 0,26, linke Wrist-Cam 0,43) stammen vom 4. Juni unter
Isaac Sim 4.x und sagen über den heutigen Stand nichts.

### Schritt 2 — Domain-Gap neu messen (`server_rl_run.sh gap`)

Die Junizahlen beschreiben ein Rendering, das es nicht mehr gibt: sie stammen von **vor** dem
Isaac-Sim-6.0-Port, vor dem Orientierungs-Fix (die Kameras filmten danach den Himmel) und vor der
Kalibrierung aus Iteration 13/14. Die Messung wird deshalb wiederholt — mit unverändertem Skript und
unveränderten Referenzbildern, damit der Vergleich trägt.

```bash
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh cams   # erzeugt /data/cam_dump/cam_*.png
./Simulation/server_rl_run.sh gap                    # misst dagegen
```

`gap` legt Referenzbilder und Messskript per `docker cp` in den laufenden Container — kein
Image-Rebuild und kein `clean` nötig, obwohl `/scripts` ins Image gebacken ist und
`Simulation/camera_reference/` dort gar nicht existiert. Gerechnet wird im Isaac-Sim-Kit-Python
(dort liegt `transformers`); der SigLIP-Download (~1,6 GB) landet unter `/data/hf_cache` und
überlebt damit ein `clean`. Ergebnisse: Tabelle auf stdout, dazu
`/data/cam_dump/domain_gap_results.json`.

Gemessen wird die Kosinus-Distanz der Bild-Embeddings durch `google/siglip-so400m-patch14-224` —
identisch mit GR00Ts Vision-Backbone, weil das BC-Fine-tuning mit `tune_visual=false` lief. Das
Skript stellt jeder Kamera ihren Juniwert und das Delta daneben.

| Kamera | 2026-06-04 (Isaac Sim 4.x) | Bewertung damals |
|---|---|---|
| `cam_left_high` | 0,1477 | moderat |
| `cam_right_high` | 0,2136 | groß |
| `cam_right_wrist` | 0,2491 | groß |
| `cam_left_wrist` | **0,4275** | kritisch |
| **Mittel** | **0,2595** | groß |
| Grundlinie real↔real (andere Kamera) | 0,2726 | — |

**Was die Zahl nicht kann.** Sie vergleicht ganze Bilder und enthält damit auch den Szeneninhalt:
Das Referenzbild zeigt die Arme mitten in der Aufgabe, der Kamera-Dump die Reset-Pose, die Würfel
liegen woanders und der Tisch ist ~15 cm näher (siehe oben). Ein Teil der Distanz ist also nicht
Renderqualität. Maßstab dafür ist die Grundlinie real↔real: 0,27 zwischen zwei *echten* Kameras
derselben Szene. Liegt der Real→Sim-Wert darunter, ist der Erscheinungs-Gap kleiner als der
Blickwinkelunterschied zweier realer Kameras.

**Entscheidungsregel — vorher festgelegt, damit die Zahl nicht nachträglich passend gedeutet wird:**

| Mittelwert | Konsequenz |
|---|---|
| < 0,20 | direkt weiter zu Schritt 3 (BC-Erfolgsrate), kein ViT-Eingriff |
| 0,20–0,35 | Schritt 3 trotzdem fahren, aber `TUNE_VISUAL=1` bzw. stärkere Domain-Randomisierung einplanen |
| > 0,35 | der visuelle Gap dominiert; RL auf diesem BC-Checkpoint trainiert gegen eine Policy, die die Szene nicht erkennt — erst Sehen reparieren |

Besonders zu beobachten ist `cam_left_wrist`: mit 0,4275 war sie im Juni der einzige kritische Wert
und ist zugleich die Kamera, die der Orientierungs-Fix am stärksten verändert hat.

#### Ergebnis (`runs/20260808/19`) — der Gap sitzt in der Belichtung, nicht in der Geometrie

| Kamera | jetzt | Juni | Delta | Bewertung |
|---|---|---|---|---|
| `cam_left_high` | 0,1885 | 0,1477 | +0,0408 | moderat |
| `cam_right_high` | 0,1733 | 0,2136 | −0,0403 | moderat |
| `cam_left_wrist` | 0,4284 | 0,4275 | +0,0009 | kritisch |
| `cam_right_wrist` | 0,3422 | 0,2491 | +0,0931 | groß |
| **Mittel** | **0,2831** | 0,2595 | +0,0236 | groß |
| Grundlinie real↔real | 0,2726 | 0,2726 | ±0,0000 | — |

**Die Messapparatur ist validiert.** Die Grundlinie real↔real kommt auf 0,27256725 — der Juniwert
auf vier Stellen identisch, bei anderem Container, anderem Python, anderer Isaac-Sim-Version. Die
Referenzbilder gehen also unverändert durch dieselbe Rechnung. Damit ist der Juni-Vergleich
belastbar: Unterschiede in den Real→Sim-Werten liegen an den Sim-Bildern, nicht am Messweg.

**Die Kalibrierung hat genau das getan, was sie sollte — und nichts darüber hinaus.** Der Mittelwert
der beiden Kopfkameras liegt bei 0,1809 gegen 0,1807 im Juni, also unverändert; ihre *Spreizung*
fällt von 0,066 auf 0,015, ein Faktor 4,3. Vorher schauten die beiden auf unterschiedliche Dinge
(konvergierende Stereobasis, ±8,3° Gierwinkel), jetzt sind sie ein symmetrisches Paar. Iteration
13/14 hat Geometrie repariert, nicht Erscheinung — und die Zahlen zeigen genau das.

**Der verbleibende Gap ist Überbelichtung.** Bildstatistik der vier Kamerapaare (auf 224 px, also
in der Auflösung, die der ViT sieht):

| Kamera | Helligkeit real → sim | Kontrast real → sim | Chroma real → sim | Pixel ≥ 245 in sim |
|---|---|---|---|---|
| `cam_left_high` | 130 → 194 | 59 → 65 | 3,6 → 1,7 | 1,1 % |
| `cam_right_high` | 128 → 194 | 59 → 65 | 4,7 → 1,6 | 1,1 % |
| `cam_left_wrist` | 109 → **232** | 65 → **28** | 5,3 → 2,8 | **16,1 %** |
| `cam_right_wrist` | 104 → **235** | 68 → **28** | 4,0 → **0,7** | **30,6 %** |

Die Sim ist überall 64–131 Graustufen heller als die Referenz. Bei den Kopfkameras überlebt der
Kontrast das noch (65 gegen 59), das Bild bleibt lesbar. Bei den Wrist-Kameras bricht er auf 28 ein,
und 16 % bzw. 31 % aller Pixel liegen bei ≥ 245, sind also nach Weiß abgeschnitten. `cam_right_wrist`
hat mit Chroma 0,71 praktisch keine Farbe mehr. **Die beiden Kameras mit dem größten Domain-Gap sind
exakt die beiden mit abgeschnittenen Pixeln** — das ist die Erklärung, und es ist auch der Grund,
warum die Kalibrierung hier nichts bewirken konnte: sie ändert, wohin die Kamera schaut, nicht wie
hell die Szene ist. Der Kontrastverlust ist Folge des Clippings, also durch weniger Licht umkehrbar.

**Zwei Einschränkungen, die den Juni-Vergleich betreffen:**

* **Die Frames entstanden unter zufälliger Beleuchtung.** Der Dump lief mit `DR=1`, und
  `_randomize_visuals()` würfelt die Dome-Intensität je Episode aus [1000, 3800]. Der Wert 0,2831 ist
  damit eine Stichprobe, kein Betriebspunkt. Für vergleichbare Zahlen `DR_ENABLED=0` setzen.
* **Der Szeneninhalt ist nicht derselbe wie im Juni.** Die Junimessung nutzte
  `_debug_obs_cam_*.png` aus einem laufenden Eval-Rollout (Arme mitten in der Aufgabe), heute kommen
  die Frames aus `dump_camera_poses.py` in der Reset-Pose. Das trifft vor allem `cam_right_wrist`:
  in der Reset-Pose sieht sie Hand und leeren Tisch, im Referenzbild die Würfel. Ein Teil der
  +0,093 ist deshalb Inhalt, nicht Belichtung. Innerhalb *eines* Sweeps entfällt der Effekt, weil
  alle Stufen dieselbe Pose zeigen.

**Entscheidung nach der vorab festgelegten Regel:** 0,2831 liegt im Band 0,20–0,35, also Schritt 3
fahren und einen visuellen Eingriff einplanen. Als Eingriff kommt zuerst die Belichtung dran, nicht
`TUNE_VISUAL=1`: der Hebel existiert bereits (`RL_DOME_INTENSITY`), die Messung dauert einen
Dump-Lauf, und sie trifft genau die zwei Kameras, die durchfallen — ein ViT-Retraining kostet ein
Vielfaches und würde dem Modell beibringen, mit weißgeclippten Bildern zu leben, statt sie zu
vermeiden.

```bash
DR_ENABLED=0 RL_DOME_SWEEP=1000,500,200,80 ./Simulation/server_rl_run.sh cams
./Simulation/server_rl_run.sh gap     # misst Basis + alle Sweep-Stufen in einem Lauf
```

`gap` erkennt die `__dome<wert>`-Varianten selbst und stellt sie als Tabelle nebeneinander, inklusive
der Stufe mit dem kleinsten Gap. Zielgröße: Wrist-Helligkeit von ~233 auf ~105 und der Anteil
geclippter Pixel gegen 0. Bleibt der Gap auch bei der besten Stufe über 0,35, ist es nicht die
Belichtung, und dann ist `TUNE_VISUAL=1` dran.

#### Sweep-Ergebnis (`runs/20260808/20`) — Belichtung ist NICHT der Hebel

Der Sweep lief mit `DR=0` (feste Beleuchtung) über 2000 → 1000 → 500 → 200 → 80, also einen
Faktor 25:

| Variante | left_high | right_high | left_wrist | right_wrist | Mittel |
|---|---|---|---|---|---|
| Basis (2000) | 0,1296 | 0,1586 | 0,4493 | 0,3098 | 0,2618 |
| dome1000 | **0,1154** | **0,1375** | 0,4553 | 0,2973 | 0,2514 |
| dome500 | 0,1346 | 0,1440 | 0,4472 | 0,2968 | 0,2556 |
| dome200 | 0,1424 | 0,1477 | 0,4254 | 0,2796 | 0,2488 |
| dome80 | 0,1450 | 0,1544 | **0,4169** | **0,2776** | **0,2485** |

**Die Hypothese ist widerlegt.** 25-fach weniger Licht bringt 0,0133 — 5 %. Die Helligkeit wurde
dabei nachweislich getroffen: die Wrist-Kameras fallen von 219 auf 119 Graustufen (real 108,5), die
Kopfkameras liegen bei `dome500` mit 131,6 exakt auf dem Referenzwert 130,3, und Clipping
verschwindet komplett. Der Hebel *wirkt*, er bewegt den Gap nur nicht. Noch deutlicher: die
Kopfkameras werden bei `dome1000` **besser**, obwohl sie dort mit 152 zu hell sind, und bei
`dome500` schlechter, obwohl die Helligkeit dort stimmt. Gap und Helligkeitstreffer laufen nicht
einmal in dieselbe Richtung. Damit ist Belichtung als Erklärung erledigt — sie war die naheliegende
Vermutung aus der Pixelstatistik und hat der Messung nicht standgehalten.

#### Was es stattdessen ist: Albedo und Hintergrund

Der Blick auf die Bilder beantwortet es sofort:

| | Referenz (real) | Sim |
|---|---|---|
| DEX3-Hand | **schwarz**, glänzend, Schrauben und Kanten sichtbar | **weiß**, mattes Plastik ohne Details |
| Unterarm | silbern/metallisch | weiß |
| Hinter dem Tisch | heller Laborboden, weiße Wand | **schwarzes Raster** (Isaac-Default-Boden) |

Das erklärt jede gemessene Zahl:

* **Wrist-Kontrast 44 statt 65.** Die Hand füllt den Großteil des Wrist-Bildes. Real ist das
  schwarz auf weißem Tisch — maximaler Kontrast. In der Sim weiß auf weiß. Kein Dimmen der Welt
  ändert ein Verhältnis: dunkler wird beides gleichzeitig.
* **Chroma 1,0–2,0 statt 3,6–5,3.** Weißer Roboter, weißer Tisch, graues Dome-Licht — die Szene ist
  fast unbunt, bis auf drei Würfel.
* **Der Gap ist über 25-fache Beleuchtung stabil.** Albedo ist eine Materialeigenschaft, keine
  Beleuchtungsfrage. Genau das erwartet man, wenn die Ursache in der Oberfläche liegt.
* **Der schwarze Rasterboden** füllt in den Kopfkameras ~45 % und in den Wrist-Kameras ~25 % des
  Bildes — eine große, kontrastreiche Fläche, die im Referenzbild schlicht nicht vorkommt.

Zwei Hebel dafür, beide ungesetzt wirkungslos (nichts ändert sich still):

| Hebel | Beispiel | Wirkung |
|---|---|---|
| `BLACK_HANDS=1` (Default) | — | `server_rl_run.sh` erzeugt und benutzt `g1_dex3_blackhands.usd` |
| `RL_GROUND_COLOR` | `0.35,0.35,0.36` | hellt den Isaac-Default-Boden auf |

```bash
DR_ENABLED=0 RL_GROUND_COLOR=0.35,0.35,0.36 ./Simulation/server_rl_run.sh cams
./Simulation/server_rl_run.sh gap
```

**Erwartung, damit sie prüfbar bleibt:** Wrist-Kontrast von 44 Richtung 65 und `cam_left_wrist`
unter 0,35. Passiert das nicht, ist auch Albedo nicht die Erklärung, und dann ist `TUNE_VISUAL=1`
an der Reihe — dann ist es Textur und Materialcharakteristik, und dagegen hilft nur, dem ViT die
Sim-Optik beizubringen.

**Einordnung, die dabei nicht untergehen soll:** Der Mittelwert liegt mit 0,2618 (bzw. 0,2485 bei
`dome80`) **unter** der Grundlinie real↔real von 0,2726. Im Schnitt ist ein Sim-Bild seinem realen
Gegenstück also näher, als zwei *echte* Kameras derselben Szene einander sind. Das Problem ist nicht
das Mittel, sondern die Verteilung: Kopfkameras 0,13–0,16 (unauffällig), `cam_left_wrist` 0,42.

#### Ergebnis (`runs/20260808/21`) — halb gemessen, und die Hälfte war schon gelöst

Der Lauf sollte beide Albedo-Hebel prüfen. Gemessen wurde nur einer:

```
[Env] Bodenfarbe -> (0.75, 0.73, 0.7) (RL_GROUND_COLOR)
[DR] Handfarbe -> (0.05, 0.05, 0.05) an 0 Materialien (RL_HAND_COLOR).
[DR] WARNUNG: kein Hand-Material getroffen — Bindungen prüfen
```

**Der Boden allein macht es nicht besser, sondern schlechter.** Gap-Mittel 0,2618 → 0,2655,
`cam_left_high` 0,1296 → 0,1542, `cam_right_wrist` 0,3098 → 0,3397; nur `cam_left_wrist` fällt
(0,4493 → 0,4113). Die Pixelstatistik sagt, warum: Der Wert war zu hell gewählt und hat den
Kontrast von einer Überschreitung in eine Unterschreitung gekippt.

| Kopfkameras | real | Lauf 20 (schwarzer Boden) | Lauf 21 (Boden 0,75) |
|---|---|---|---|
| Helligkeit | 130,3 | 169,5 | 203,3 |
| Kontrast | 59,4 | 79,3 | **39,4** |
| Chroma | 9,1 | 2,7 | **23,3** |

Ein Zwischenwert (`0.35,0.35,0.36`) ist der nächste Versuch — der schwarze Rasterboden bleibt
falsch, 0,75 war nur die falsche Richtung von zu wenig zu zu viel.

**Der Hand-Hebel konnte nicht funktionieren, und er war überflüssig.** Zwei Gründe:

1. *Er kann es nicht.* Die `/visuals`-Prims des Roboters sind `instanceable`; die Meshes liegen in
   USD-Prototypen. `Usd.PrimRange` läuft dort nicht hinein, und eine Bindung am Instance-Root
   komponiert nicht in den Prototyp. Daher 0 Materialien. Der Prim-Pfad-Filter war zusätzlich
   wirkungslos: der Roboter-Root heißt `g1_29dof_with_hand_rev_1_0` und enthält `_hand_` selbst,
   also passte *jeder* Pfad darunter.
2. *Es gab ihn schon.* [`recolor_hands_black.py`](../../Simulation/g1_dex3_sim/recolor_hands_black.py)
   löst seit Juni exakt dieses Problem — de-instanziert die Hand-`/visuals` und bindet je Mesh mit
   `strongerThanDescendants`. `data/g1_dex3_blackhands.usd` liegt seit dem 04.06. im Repo,
   `g1_dex3_cfg.py` zeigt per Default darauf, `kisski_sim_submit.sh` erzeugt es automatisch.

Warum die Hände auf dem Server trotzdem weiß sind: `server_rl_run.sh` setzte
`ASSET_PATH=$CHECKPOINT_PATH/g1_dex3.usd` — das Original. Der KISSKI-Launcher zog das
schwarzhändige Asset, dieses Skript nicht. Behoben: `BLACK_HANDS=1` ist Default, `ensure_black_hands`
erzeugt das Asset bei Bedarf im Container und fällt bei Fehlschlag aufs Original zurück. Der
Laufzeit-Hebel `RL_HAND_COLOR` ist entfernt — ein zweiter Mechanismus für dieselbe Sache, der
nachweislich nicht greift.

**Zwei Werkzeugfehler, die der Lauf offengelegt hat:**

* Die Sweep-Zeilen in Lauf 21 sind **byteidentisch mit Lauf 20** (`md5sum` geprüft) — alte
  `__dome*.png` überlebten im Container, obwohl `sweep=<aus>` lief. `cams` löscht sie jetzt vorher.
* Das Sweep-Mittel wurde über *vorhandene* Kameras gebildet und gegen ein 4-Kamera-Mittel gestellt.
  Deshalb stand da „Keine Sweep-Variante schlägt die Basis" — auf denselben zwei Kameras gerechnet
  war es umgekehrt. `measure_domain_gap.py` vergleicht jetzt ein Delta gegen dieselben Kameras und
  markiert unvollständige Zeilen mit `*`.

**Stand der Vorhersage:** Das Kriterium (`cam_left_wrist` < 0,35) ist mit 0,4113 nicht erfüllt — der
Test der eigentlichen Hypothese hat aber nie stattgefunden. Damit daraus kein Nachbessern bis zum
Erfolg wird: **ein** Lauf mit schwarzen Händen und Boden 0,35. Bleibt `cam_left_wrist` dann über
0,35, ist Albedo widerlegt und `TUNE_VISUAL=1` die Konsequenz — ohne weiteren Zwischenversuch.

#### Ergebnis (`runs/20260808/22`) — Albedo bestätigt, Kriterium um 0,0056 verfehlt

Diesmal liefen beide Hebel, nachweisbar im Log:

```
OK: schwarzes Material an 16 Hand-/visuals-Roots gebunden (de-instanziert, strongerThanDescendants)
Asset erzeugt: /data/checkpoints/groot-g1dex3-checkpoint/g1_dex3_blackhands.usd
[Env] Bodenfarbe -> (0.35, 0.35, 0.36) (RL_GROUND_COLOR)
```

| Kamera | Juni | Lauf 20 | Lauf 21 | **Lauf 22** | Δ zu Juni |
|---|---|---|---|---|---|
| `cam_left_high` | 0,1477 | 0,1296 | 0,1542 | **0,1121** | −0,0356 |
| `cam_right_high` | 0,2136 | 0,1586 | 0,1567 | **0,1310** | −0,0826 |
| `cam_left_wrist` | 0,4275 | 0,4493 | 0,4113 | **0,3556** | −0,0719 |
| `cam_right_wrist` | 0,2491 | 0,3098 | 0,3397 | **0,2930** | +0,0439 |
| **MITTEL** | 0,2595 | 0,2618 | 0,2655 | **0,2229** | −0,0366 |

Alle vier Kameras verbessern sich gegenüber Lauf 21, drei von vier gegenüber Juni. Das Mittel liegt
mit 0,2229 unter der Grundlinie real↔real (0,2726) und praktisch auf der sim-internen Streuung
(0,2163) — real→sim ist damit das 0,8-fache des real→real-Abstands.

Die Pixelstatistik zeigt, dass der Mechanismus der vermutete war. Der Kontrast der Kopfkameras
trifft jetzt den realen Wert fast exakt, nachdem er in Lauf 20 zu hoch und in Lauf 21 zu niedrig war:

| Kopfkameras | real | Lauf 20 | Lauf 21 | **Lauf 22** |
|---|---|---|---|---|
| Helligkeit | 130,3 | 169,5 | 203,3 | 184,9 |
| Kontrast | 59,4 | 79,3 | 39,4 | **59,0** |
| Chroma | 9,1 | 2,7 | 23,3 | 19,7 |

Der Wrist-Kontrast, der Auslöser der ganzen Hypothese, geht von 36,1 (Lauf 20) auf **68,9** bei real
65,0 — die schwarze Hand auf hellem Tisch stellt genau das Verhältnis her, das im Datensatz steht.

**`RL_GROUND_COLOR` macht den Boden nicht grau, sondern entdimmt Isaacs blaue Rastertextur.**
Ein nahezu neutraler Wert (0,35/0,35/0,36) rendert sichtbar blau, und die Chroma steigt vom
Isaac-Default 2,7 auf 19,7 bei real 9,1. Ein neutraler Eingang kann kein farbiges Ergebnis
erzeugen, wenn er die Albedo direkt setzt — er wirkt also als Tint auf eine Textur (vermutlich
`Looks/theGrid.inputs:diffuseColor` des Default-Bodens; in der Quelle nicht geprüft, Isaac Lab liegt
nur im Container). Praktische Folge: Aufhellen des Bodens erhöht zwangsläufig die Sättigung. Für
einen wirklich neutralen Boden müsste das Material ersetzt statt getönt werden. Betroffen sind vor
allem die Kopfkameras — und die sind mit 0,11/0,13 ohnehin unauffällig.

**Stand der Vorhersage:** `cam_left_wrist` liegt bei **0,3556**, das Kriterium war **< 0,35**. Um
0,0056 verfehlt. Das wird hier nicht gerundet: Albedo erklärt einen großen, messbaren Teil des Gaps
— es reicht aber nicht, um die Wrist-Kamera aus der Ausreißerrolle zu holen (sie ist weiterhin das
1,3-fache der real↔real-Grundlinie). Nach der Regel gilt damit: **kein weiterer Albedo-Versuch**,
`TUNE_VISUAL=1` ist die Konsequenz.

Bevor dafür ein H100-Lauf über ~48 h gebucht wird, ist Schritt 3 (BC-Erfolgsrate in der Sim) fällig
— er stand ohnehin als Nächstes an, kostet eine Sim-Eval statt eines Trainings und misst die
Größe, die `TUNE_VISUAL=1` verbessern soll, direkt statt über den Proxy. Ist die Erfolgsrate 0,
ist die Entscheidung bestätigt. Das ist ausdrücklich **kein** weiterer Renderversuch.

Offen und in derselben Kamera wirksam: die bekannte Abweichung von ~15 cm im Roboter-Tisch-Abstand.
Der Modulkopf von `measure_domain_gap.py` warnt selbst davor, dass die Zahl Szeneninhalt enthält;
die Wrist-Kamera ist dafür die empfindlichste. Das ist für Schritt 3 ohnehin zu klären, weil eine
unerreichbare Tischplatte jede Erfolgsrate auf 0 nagelt, unabhängig von der Optik.

### Schritt 3 — BC-Erfolgsrate in der Sim (`server_rl_run.sh eval`)

Der Domain-Gap ist ein Proxy. Diese Zahl ist die Zielgröße selbst: **schafft die BC-Policy die
Aufgabe in der Simulation überhaupt?** Sie entscheidet zwei Dinge auf einmal — ob sich ein
`TUNE_VISUAL=1`-Lauf über ~48 h lohnt, und was der Nullpunkt für jeden späteren RL-Vergleich ist.
Ohne sie lässt sich „RL hat geholfen" nicht belegen, egal wie die RL-Kurve aussieht.

```bash
HF_TOKEN=hf_... NUM_EPISODES=20 EPISODE_LENGTH_S=120 DR_ENABLED=0 \
    ./Simulation/server_rl_run.sh eval
```

Das Inferenz-Backend wird ausdrücklich über `GROOT_INFERENCE_BACKEND` gewählt: `eager`
(Default und Referenz), `compile` (`torch.compile`) oder `tensorrt` (vorher einmal
`./Simulation/server_rl_run.sh optimize all`). Beispiel für den schnellen Modus:

```bash
GROOT_INFERENCE_BACKEND=tensorrt CAMERA_RENDER_EVERY_N=8 SCENE_CAM=0 \
    NUM_EPISODES=20 EPISODE_LENGTH_S=120 DR_ENABLED=0 \
    ./Simulation/server_rl_run.sh eval
```

Alle Voraussetzungen, Backend-Vergleiche und der automatische Engine-Pfad stehen in
[inferenz-optimierung.md](../simulation/inferenz-optimierung.md). Für methodisch vergleichbare
BC-/RL-Nullpunkte Backend und Renderkonfiguration im Ergebnisbericht festhalten.

Was hier läuft, ist **nicht** `rl_finetune.py`, sondern die vollständige Closed-Loop-Pipeline
(`entrypoint_sim.sh`: GR00T-Policy-Server über ZMQ + Isaac-Lab-Client). Der Unterschied ist für die
Zahl wesentlich: der Client führt 8 Schritte eines 16er-Chunks aus, während der RL-Rollout jeden
Step neu plant und 15 von 16 Vorhersagen wegwirft. Gemessen werden soll der Betriebsmodus.

| Parameter | Wert | Begründung |
|---|---|---|
| `NUM_EPISODES` | 20 | Auflösung siehe unten |
| `EPISODE_LENGTH_S` | 120 | 3× die menschliche Demo. `replay_episode0.npz` hat 1173 Steps = 39 s bei 30 Hz. Der cfg-Default von 300 s ist das 7,7-fache und kostet nur Laufzeit |
| `EXECUTION_HORIZON` | 8 (Default) | wie im Deployment |
| `DR_ENABLED` | 0 | feste Beleuchtung, sonst ist jede Episode eine andere Szene |

Laufzeit: bis zu 20 × 120 s × 30 Hz = 72 000 Env-Steps mit 5 gerenderten Kameras, dazu ~9000
Policy-Aufrufe. Grobe Schätzung 1–2 h; erfolgreiche Episoden brechen früher ab. Ergebnisse landen in
`$HOST_DATA_DIR/sim_results/results.json`, Videos je Episode in `$HOST_DATA_DIR/sim_videos/`.

**Entscheidungsregel — vorher festgelegt, damit die Zahl nicht nachträglich gedeutet wird:**

| Erfolgsrate | Konsequenz |
|---|---|
| **0/20** | Die Policy löst die Aufgabe in der Sim nicht. **Erst Geometrie ausschließen** (s. u.), dann ist `TUNE_VISUAL=1` bestätigt — RL auf einem Nullpunkt-Reward kann nichts lernen, dem fehlt das Startsignal |
| **1–3/20** | Schwaches, aber echtes Signal — der beste Startpunkt für RL. Kein `TUNE_VISUAL`-Lauf, direkt RL, denn FPO braucht genau diesen seltenen Erfolg als Gradient |
| **> 3/20** | BC funktioniert in der Sim. RL ist reine Verbesserung, `TUNE_VISUAL` erübrigt sich |

**Was 0/20 statistisch heißt.** Bei 20 Episoden und null Erfolgen liegt die obere 95-%-Grenze der
wahren Rate bei rund 14 %. „0/20" schließt also eine schwache Fähigkeit nicht aus, es schließt nur
eine brauchbare aus. Für die Unterscheidung 0 % gegen 5 % wären ~60 Episoden nötig — das ist erst
interessant, wenn überhaupt ein Erfolg auftritt.

**Vor der Interpretation zu prüfen — sonst misst der Lauf etwas anderes als gedacht:**

* **Erreicht die Hand den Tisch?** Die bekannte Abweichung von ~15 cm im Roboter-Tisch-Abstand
  nagelt jede Erfolgsrate auf 0, unabhängig von der Optik. Die Videos in `sim_videos/` zeigen das
  unmittelbar: greift der Roboter ins Leere oder daneben, ist das Ergebnis kein Aussagewert über das
  Sehen. **Dieser Punkt ist die einzige zulässige Erklärung für eine 0, die nicht `TUNE_VISUAL=1`
  auslöst** — und er ist vor dem Lauf offen, nicht danach erfunden.
* **Sieht die Policy etwas?** Der Runner legt in Episode 1/Step 0 `_debug_obs_cam_*.png` in
  `sim_videos/` ab — vier Bilder, genau die Modell-Eingabe. Sind die leer, weiß oder schwarz, ist
  der Lauf ungültig und keine Aussage über die Policy.

#### Zwischenstand (runs/20260808/23, erste 2 von 20 Episoden)

Der Lauf startete 16:19 Uhr mit genau den Werten oben; das Log wurde bei Episode 3 kopiert. Beide
fertigen Episoden: **misslungen, volle 3600 Steps**, kein früher Abbruch. Beide Vorprüfungen sind
damit beantwortet — und keine der beiden erklärt das Ergebnis:

* **Die Policy sieht die Szene.** `_debug_obs_cam_left_high.png` zeigt Tisch, alle drei Würfel und
  beide schwarzen Hände scharf und mittig. Kein leeres oder überstrahltes Bild.
* **Die Hand erreicht den Tisch.** Im Szenenvideo liegt die linke Hand über weite Strecken auf
  Tischhöhe direkt neben dem roten Würfel. Die 15-cm-Abweichung äußert sich **nicht** als
  „greift ins Leere". Damit fällt die einzige zugelassene Nicht-Sehen-Erklärung für eine 0 weg.

Der eigentliche Befund steht aber nicht im Erfolgszähler, sondern im Video: **die Würfel bewegen
sich in 7200 Steps kein einziges Mal.** Anfangs- und Endbild beider Episoden zeigen sie
pixelgenau an derselben Stelle (die Startlagen sind zwischen den Episoden randomisiert, die
Endlagen also nicht durch eine feste Szene erklärt). Die Arme bewegen sich dabei durchgehend: die
Frame-zu-Frame-Differenz im Tischausschnitt liegt konstant bei 0,35–0,41 über alle sechs
20-s-Fenster, ohne jede Phasenstruktur. Kein Anfahren, kein Greifen, kein Anheben — eine
gleichförmige Bewegung, die den Würfel nie berührt.

Das ist informativer als die Erfolgsrate und war mit dem bisherigen Instrumentarium nur per Auge
am Video zu sehen. Deshalb protokolliert der Runner seit diesem Lauf zwei Zahlen je Episode
(`env.get_reach_diagnostics()`, kostet keinen zusätzlichen Render-Pass):

| Feld in `results.json` | Bedeutung |
|---|---|
| `min_reach_m` / `min_reach_step` | kleinster Abstand Hand↔Würfel in der Episode, und wann |
| `block_shift_max_m` | größte Verschiebung eines Würfels gegenüber dem Reset-Layout |

Damit trennt der nächste Lauf die beiden Lesarten quantitativ: kommt die Hand auf wenige
Zentimeter heran und greift trotzdem nicht, ist es Wahrnehmung/Politik und `TUNE_VISUAL=1` steht;
bleibt der Abstand groß, ist es Geometrie und ein ViT-Lauf wäre verschwendet.

#### Erste Messung (runs/20260808/24) — und warum sie so noch nichts entscheidet

Zwei Episoden à 40 s (1200 Steps), 0/2, mit aktiver Diagnose:

| Episode | `min_reach_m` | `min_reach_step` | `block_shift_m` (drei Würfel) |
|---|---|---|---|
| 1 | 0,1476 | 13 | 0,0189 / 0,0165 / 0,0200 |
| 2 | 0,1473 | 18 | 0,0209 / 0,0140 / 0,0200 |

Die naheliegende Lesart — „15 cm Abstand, genau die bekannte Tischabweichung, also Geometrie" —
hält der Prüfung **nicht** stand. Beide Spalten messen etwas anderes als gedacht:

* **Der Abstand war am falschen Ende der Hand gemessen.** `_get_hand_positions()` liefert
  `left/right_wrist_yaw_link`, also die Handwurzel. Laut URDF liegen zwischen ihr und dem letzten
  Fingergelenk 0,0415 + 0,0777 + 0,0458 = **16,5 cm**, die Fingerspitze noch etwas weiter. Ein
  Handwurzel-Abstand von 14,7 cm liegt damit *innerhalb der eigenen Handgeometrie* und ist
  mehrdeutig: er beschreibt „Würfel liegt in der Greiföffnung" genauso gut wie „Würfel 15 cm
  daneben". Die Zahl kann die Frage, für die sie eingebaut wurde, in dieser Form nicht beantworten.
* **Die Würfel-Verschiebung misst den Solver, nicht den Roboter.** Alle drei Würfel verschieben
  sich um 1,4–2,1 cm, in beiden Episoden, auch die, in deren Nähe nie eine Hand war (der dritte
  in beiden Episoden auf 4 Stellen identisch: 0,0200). Das ist das Einschwingen des Kontakts nach
  dem Reset. Das Kriterium „>1 cm = angefasst" meldete deshalb 2/2 — ein reiner Fehlalarm.

Belastbar ist dagegen `min_reach_step`: **13 bzw. 18 von 1200.** Die größte Annäherung der
gesamten Episode fällt in die erste halbe Sekunde und wird in den folgenden 40 s nie wieder
unterboten. Daraus schien zu folgen, dass sich die Arme nie auf einen Würfel zubewegen — **auch
das war ein Artefakt des Messpunkts** (s. Lauf 25): die Handwurzel bleibt zurück, während sich die
Finger nach vorn strecken, ihr Minimum liegt deshalb früh. Von Lauf 24 bleibt am Ende nur, dass
seine Zahlen die Frage nicht beantworten konnten.

Konsequenz im Code (alles in `get_reach_diagnostics` / `run_g1_dex3_sim_eval.py`):

| Änderung | Grund |
|---|---|
| Messkörper = Fingerspitzen (`*_hand_index_1_link`, `*_middle_1_link`, `*_thumb_2_link`), Fallback Handfläche → Handwurzel | nur an der Kontaktfläche heißt „klein" auch „am Würfel" |
| `reach_frame` in `results.json` + Startzeile | die Zahl darf nie ohne ihren Bezugsrahmen gelesen werden |
| `reach_start_m` zusätzlich zu `min_reach_m` | erst die Differenz zeigt, ob sich der Roboter überhaupt angenähert hat |
| Würfel-Grundlage erst nach 30 Steps Karenz | schneidet das Einschwingen ab, das sonst als „angefasst" zählt |
| `finger_span_max_rad` (Action-Dims 14:28) | zeigt, ob überhaupt ein Griff kommandiert wurde |

#### Die eigentliche Messung (runs/20260808/25) — die Hand ist am Würfel

> **Nachtrag Lauf 28:** die „Fingerspitzen"-Abstände unten sind ab dem *distalen Gelenk* gemessen,
> die Kuppe liegt 5,2 cm weiter. Alle Zahlen sind also um diesen Betrag zu groß — die Schlussfolgerung
> „die Hand ist am Würfel" wird dadurch stärker. Details und Fix im Abschnitt zu Lauf 28.

Derselbe Lauf mit korrigiertem Messpunkt, wieder 2 × 40 s, wieder 0/2:

| Episode | `reach_start_m` → `min_reach_m` | `min_reach_step` | `block_shift_m` |
|---|---|---|---|
| 1 | 0,135 → **0,0396** | 58 | 0,0147 / 0,0 / 0,0 |
| 2 | 0,127 → **0,0420** | 1014 | 0,0570 / 0,0 / 0,0 |

Das kehrt den Befund aus Lauf 24 um:

* **Der Roboter nähert sich.** 9,5 cm Annäherung gegenüber der Startpose, in beiden Episoden auf
  den Millimeter gleich. Das ist kein Zufallsprodukt einer wackelnden Bewegung.
* **Die Fingerspitzen erreichen den Würfel.** 4,0 bzw. 4,2 cm zum Würfel*mittelpunkt*, bei 5 cm
  Kantenlänge also rund **1,5 cm zur Oberfläche**.
* **Es gibt echten Kontakt.** Genau ein Würfel verschiebt sich (1,5 bzw. 5,7 cm), die beiden
  anderen exakt 0,0 — die Karenzzeit funktioniert, und der bewegte Würfel wurde angefasst.

Damit ist die Vorprüfung „Geometrie" endgültig erledigt: die Hand kommt hin, berührt den Würfel
und schiebt ihn. Was fehlt, ist der Griff. Dafür bleiben zwei Erklärungen, und die vorregistrierte
Regel („in Reichweite, greift nicht → `TUNE_VISUAL=1`") unterscheidet sie nicht:

1. **Die Politik versucht keinen Griff** — die Finger bleiben starr, der Würfel wird nur
   angestoßen. Das wäre Wahrnehmung/Politik und `TUNE_VISUAL=1` wäre richtig.
2. **Der Griff rutscht** — die Finger schließen, aber Reibung/Kontaktparameter halten den Würfel
   nicht. Das wäre Sim-Physik, und ein 48-h-ViT-Lauf wäre verschwendet.

Ein 5,7-cm-Schub spricht eher für (1), beweist es aber nicht. Deshalb protokolliert der Runner
zusätzlich `finger_span_max_rad`: die größte Spannweite, die ein Fingergelenk-**Kommando**
(Action-Dims 14:28) über die Episode durchläuft. Nahe 0 heißt „die Finger wurden nie bewegt" und
entscheidet (1) direkt am Kommando, noch vor jeder Physikfrage.

**Nächster Schritt** — beides, in dieser Reihenfolge, zusammen unter 15 Minuten:

```bash
# a) Greif-Physik isoliert: Würfel exakt an die aufgezeichneten Greifpunkte, echte
#    Dataset-Aktionen, kein Modell. Wird ein Würfel angehoben?  (neue Aktion 'grasp')
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh grasp

# b) Eval erneut, jetzt mit Fingerspur
HF_TOKEN=hf_... NUM_EPISODES=2 EPISODE_LENGTH_S=40 DR_ENABLED=0 \
    ./Simulation/server_rl_run.sh eval
```

Hebt (a) den Würfel (`max_cube_lift_cm` > ~2) und ist die Fingerspanne in (b) klein →
Politik/Wahrnehmung, `TUNE_VISUAL=1` steht. Hebt (a) nichts → zuerst die Kontaktparameter, ViT
später. Zum Kontext: die Finger-Positionsgrenzen waren schon einmal die Ursache eines
fallengelassenen Würfels; `_widen_finger_joint_limits()` weitet sie seit Juni auf die echte
Dataset-Range, dieser Pfad ist also bereits abgedeckt.

#### Antwort (runs/20260808/26): der Griff scheitert auch ohne Modell

`grasp` beantwortet (a) eindeutig — und zwar in die zweite Richtung:

| Größe | Wert | Lesart |
|---|---|---|
| Arm-Tracking, mittel | **0,019 rad** | der Sim führt die aufgezeichneten Aktionen sauber aus (Schwelle 0,1) |
| worst arm joints | 0,10–0,12 rad | keine zu schwachen PD-Gains |
| min Handfläche→Würfelmitte | 6,8 cm | die Hand ist am Würfel |
| **max Würfel-Anhebung** | **0,0 cm** | **kein Würfel wird angehoben** |

Aufbau: echte Dataset-Aktionen, kein Modell, kein Server, und die Würfel per `--grasp-test`
exakt an den aufgezeichneten Greifpunkten. Unter diesen Bedingungen ist die Politik vollständig
aus der Kette entfernt — **und der Griff scheitert trotzdem.**

Damit ist entschieden:

* **Es liegt nicht am Modell und nicht an der Wahrnehmung.** Der Fehler reproduziert sich ohne
  Modell in der Schleife. `TUNE_VISUAL=1` ist vorerst vom Tisch; ein 48-h-ViT-Lauf würde gegen ein
  Ziel trainieren, das die Sim physikalisch nicht hergibt.
  *(Eingeschränkt durch Lauf 27, siehe unten: die Würfel-Positionen dieses Tests sind geschätzt,
  ein ausbleibendes Anheben kann auch an der Platzierung liegen. Die Reihenfolge — erst Sim, dann
  ViT — bleibt davon unberührt.)*
* **Die BC-Erfolgsrate misst derzeit nicht die Policy.** Die 0/2 aus Lauf 25 sind kein Befund über
  den Checkpoint. Schritt 3 taugt als „Nullpunkt für jeden RL-Vergleich" erst, wenn ein Würfel
  überhaupt angehoben werden kann.
* **RL wäre in diesem Zustand wirkungslos.** Sowohl `reward_mode=binary` als auch die
  `stack`/`height`-Terme des Shaped-Reward setzen ein Anheben voraus. FPO bekäme aus diesen
  Komponenten nie ein Signal — der Lauf liefe, ohne lernen zu können. Das erklärt den bisher
  unbewiesenen Lerneffekt zwanglos.

Erste Spur, noch nicht beweisend: der tiefste **Handflächen**punkt liegt bei z = 0,953 (links)
bzw. 0,943 (rechts), die Würfel-Oberkante bei 0,940 — die Handflächen bleiben also über dem
Würfel. Da die Finger von der Handfläche aus nach vorn zeigen, ist damit noch nicht gesagt, ob sie
die Seiten umschließen. (Die Zeile war bis Lauf 26 als „Würfel-Oberseite z≈0,915" beschriftet,
tatsächlich ist das die Würfel-*Mitte* — dieselbe Sorte Beschriftungsfehler wie bei der Handwurzel
in Lauf 24, jetzt korrigiert.)

Deshalb misst `grasp` ab sofort zusätzlich, ob die Finger überhaupt schließen:

| Feld in `sim_results_replay/results.json` | Bedeutung |
|---|---|
| `mean_finger_tracking_error_rad` | folgen die Fingergelenke ihrem Kommando? |
| `finger_span_commanded_rad` / `finger_span_achieved_rad` | Greifbewegung kommandiert vs. tatsächlich gefahren |

**Nächster Schritt:** `grasp` noch einmal (rund 5 Minuten, kein Modell nötig). Bleibt
`finger_span_achieved_rad` deutlich hinter `finger_span_commanded_rad` zurück, klemmt eine
Gelenkgrenze die Greifbewegung ab und `_widen_finger_joint_limits()` greift nicht wie gedacht.
Sind beide Spannen groß, schließen die Finger — dann liegt es an Kontakt/Reibung oder daran, dass
der Würfel nicht zwischen den Fingern liegt. Gegen reine Reibung spricht dabei die Konfiguration:
die Würfel wiegen 50 g bei `static_friction=3.0` / `dynamic_friction=2.5`.

#### Nachmessung (runs/20260808/27): die Finger schließen — die Gelenkgrenze ist es nicht

| Größe | Wert | Lesart |
|---|---|---|
| Finger-Tracking, mittel | 0,043 rad | die Fingergelenke folgen ihrem Kommando |
| Fingerspanne kommandiert / erreicht | **2,09 / 2,10 rad** | die Greifbewegung wird vollständig gefahren |
| Finger-Tracking, max | 0,77 rad | einzelner Ausschlag — Kontakt oder Transiente, nicht dauerhaft |
| min Handfläche→Würfelmitte | 6,8 cm | unverändert |
| max Würfel-Anhebung | 0,0 cm | unverändert |

Damit ist der erste Zweig der Entscheidungsregel erledigt: `erreicht ≥ kommandiert`, also klemmt
**keine Gelenkgrenze** die Greifbewegung ab, und `_widen_finger_joint_limits()` arbeitet wie
vorgesehen. Die Hand öffnet und schließt über volle 2,1 rad.

Übersehen wurde bis hierher die aussagekräftigste Zeile des Laufs — die Würfel-XY am Ende:

| Würfel | gesetzt | am Ende | verschoben |
|---|---|---|---|
| links | (0,35 / 0,20) | (0,35 / 0,17) | 3,0 cm |
| rechts | (0,37 / −0,16) | (0,37 / −0,11) | 5,0 cm |
| Mitte | (0,35 / 0,00) | (0,37 / 0,03) | 3,6 cm |

Alle drei werden **angefasst und weggeschoben**, keiner wird angehoben — in Lauf 26 und 27
millimetergleich. Das ist die Signatur einer Hand, die den Würfel im Vorbeifahren wegstößt, nicht
die einer Hand, aus der er herausrutscht.

**Korrektur zu Lauf 26.** Die dortige Formulierung „der Griff scheitert auch ohne Modell" ist
schwächer, als sie klingt. Die Würfel-Positionen des Greif-Tests sind eine *Schätzung*
(tiefster Handflächenpunkt + 5 cm in +x, hart kodiert in
[`run_g1_dex3_replay.py`](../../Simulation/g1_dex3_sim/run_g1_dex3_replay.py)); das Dataset
speichert keine Objekt-Posen. Der Würfel liegt außerdem die ganze Episode dort, obwohl die Hand
den Punkt nur einmal passiert. Ein ausbleibendes Anheben kann deshalb genauso gut an Ort und
Zeitpunkt der Platzierung liegen wie an der Greif-Physik. Belastbar aus Lauf 26/27 bleibt: Arme
und Finger fahren die aufgezeichnete Trajektorie sauber ab, und die Würfel werden berührt.

#### Der Test ohne Platzierungs-Annahme: `GRASP_MODE=hold`

Um die beiden Erklärungen zu trennen, setzt der Replay den Würfel jetzt wahlweise **im Moment des
Zugreifens genau zwischen die drei Fingerspitzen**. Der Mittelpunkt des Fingerdreiecks *ist* die
Greiföffnung, und der Auslöser ist die Greifbewegung selbst (Öffnung fällt unter 70 % ihrer
bisher größten Weite) — Ort und Zeitpunkt sind damit per Konstruktion richtig, statt geschätzt.

```bash
HF_TOKEN=hf_... GRASP_MODE=hold ./Simulation/server_rl_run.sh grasp
```

| Feld in `sim_results_replay/results.json` | Bedeutung |
|---|---|
| `hold_ratio` | Anteil der Steps nach dem Einsetzen, in denen der Würfel < 4 cm am Fingerdreieck bleibt |
| `hold_rise_cm` | maximale Höhe über dem Einsetzpunkt — die Hand trägt ihn nach oben |
| `hold_final_z` | Endhöhe; ≈ 0,915 heißt: auf den Tisch gefallen |
| `min_fingertip_cube_dist_cm` / `min_fingertip_step` | Abstand ab den **Fingerspitzen** (nicht der Handfläche), mit Zeitpunkt |
| `finger_spread_min_cm` / `finger_close_step` | engste Greiföffnung je Hand und wann sie eintritt |

**Entscheidungsregel:** `hold_ratio` > ~0,5 → die Greif-Physik trägt, und das Problem ist die
Platzierung des Tests (bzw. im Closed Loop: die Politik trifft den Würfel nicht). `hold_ratio` ≈ 0
bei Endhöhe ≈ Tischauflage → der Würfel rutscht aus der geschlossenen Hand, dann sind Kontakt und
Reibung dran. Gegen letzteres spricht weiterhin die Konfiguration: 50 g bei `static_friction=3.0` /
`dynamic_friction=2.5`.

Unabhängig vom Modus laufen `min_fingertip_step` und `finger_close_step` ab sofort mit. Fallen die
beiden Zeitpunkte weit auseinander, greift die Hand ins Leere — dann ist die Reihenfolge
„erst Platzierung, dann Physik" ohnehin die richtige.

#### Lauf 28: der Messpunkt war zum dritten Mal falsch

`GRASP_MODE=hold` meldete für beide Hände „kein Zugreifen erkannt — Finger schließen nie unter
70 % ihrer größten Öffnung", bei einer engsten Greiföffnung von 7,4 cm (links) und 7,6 cm
(rechts). Das widerspricht der Fingerspanne aus demselben Lauf: 2,09 rad kommandiert, 2,09 rad
erreicht. Eine Hand, deren Beugegelenke 120° durchfahren, kann ihre Fingerkuppen nicht nahezu
still halten — also stimmte die Messung nicht.

Die Gegenprobe lief ohne Sim, direkt auf der mitgelieferten Aktionsdatei
[`replay_episode0.npz`](../../Simulation/g1_dex3_sim/replay_episode0.npz):

| | linke Hand | rechte Hand |
|---|---|---|
| stärkste Beugung (aufgezeichnet) | Step 330 | Step 186 |
| Beugung > 80 % | Steps 320–831 | Steps 102–203 |
| engste Greiföffnung (gemessen) | Step 36 | Step 109 |

Links liegen 294 Steps (rund 10 s) zwischen dem stärksten Zugreifen und dem, was die Diagnose als
engste Öffnung ausgab. Die Ursache steht in der URDF: der Frame eines distalen Fingerglieds sitzt
**im Gelenk**, und das Beugen dieses Gelenks **dreht den Frame nur** — sein Ursprung wandert nicht.
Die Kollisionsmesh reicht von dort noch **5,2 cm** weiter (STL-Bounding-Box, `index_1`/`middle_1`
lokal +x, `thumb_2` lokal ∓y). Gemessen wurde also das letzte Fingergelenk, und der Griff selbst
war in den Zahlen unsichtbar.

Das ist derselbe Fehler zum dritten Mal — Handwurzel (Lauf 24), Handfläche (Lauf 26), distales
Gelenk (Lauf 25/28). **Regel: ein Körper-Frame ist keine Kontaktfläche.** Wer den Abstand zu einem
5-cm-Würfel misst, darf nicht 5,2 cm vor der Kuppe anfangen.

**Was dadurch neu zu lesen ist:** die „Fingerspitzen"-Abstände aus Lauf 25 (4,0 / 4,2 cm zur
Würfelmitte) sind Abstände ab dem distalen Gelenk. Um die Kuppenlänge korrigiert liegen die echten
Kontaktflächen im Würfel — die Politik war also in Kontaktreichweite, nicht 1,5 cm davor. Die
Aussage „die Hand ist am Würfel" wird dadurch stärker, nicht schwächer; die 0,0 cm Anhebung bleibt
das Rätsel.

**Behoben in** [`g1_dex3_blockstack_env.py`](../../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py):
`get_contact_points_w()` liefert jetzt die Kuppen — lokaler Versatz, mit der Körper-Orientierung
mitgedreht (`quat_apply`), sonst zeigte er beim gebeugten Finger in die falsche Richtung. Eval,
Replay und Greif-Test ziehen ihre Kontaktpunkte aus dieser einen Quelle, damit nicht wieder drei
Stellen drei verschiedene Punkte messen. `results.json` bekommt zusätzlich
`finger_spread_max_cm`, weil „nie unter 70 % des Maximums" ohne das Maximum nicht lesbar ist.

**Nächster Schritt:** beide Messungen mit dem korrigierten Bezugspunkt wiederholen —

```bash
HF_TOKEN=hf_... GRASP_MODE=hold ./Simulation/server_rl_run.sh grasp
```

Erst wenn `finger_spread_min_cm` mit der Beugung aus der Tabelle oben zusammenfällt, misst die
Diagnose, was sie behauptet. Danach entscheidet `hold_ratio` wie oben beschrieben.

#### Lauf 29: der Würfel hebt ab, mit einem Vorbehalt

`runs/20260812/01` (2026-08-12), `GRASP_MODE=hold` mit dem korrigierten Fingerkuppen-Messpunkt.
**`max_cube_lift_cm` = 7,9** — nach vier Läufen mit 0,0 cm hebt zum ersten Mal ein Würfel ab.

Zuerst der Nachweis, dass die Messung jetzt misst, was sie behauptet (die Bedingung aus Lauf 28):

| | Lauf 28 (distales Gelenk) | Lauf 29 (Fingerkuppe) |
|---|---|---|
| engste Greiföffnung links | 7,4 cm → „kein Zugreifen erkannt" | **5,2 cm** von max. 11,0 cm |
| engste Greiföffnung rechts | 7,6 cm → „kein Zugreifen erkannt" | **4,9 cm** von max. 10,5 cm |
| min Fingerkuppe→Würfelmitte | — | **2,9 cm** (Step 338) |
| min Handfläche→Würfelmitte | 6,8 cm | 9,8 cm |

Die Hand öffnet und schließt jetzt sichtbar über 11,0 → 5,2 cm. Der Unterschied zwischen
Handflächen- und Fingerkuppen-Abstand (9,8 gegen 2,9 cm) beziffert die Messpunkt-Verschiebung,
die drei Läufe lang die Interpretation verdorben hat, auf rund 7 cm.

**Die beiden Hände erzählen unterschiedliche Geschichten:**

| | links (`block_0`, rot) | rechts (`block_1`, grün) |
|---|---|---|
| Einsetzen bei Step | 323 | 91 |
| Haltequote | **0,238** (202/849) | 0,002 (2/1081) |
| max. Anhebung über Einsetzpunkt | **+7,9 cm** | −1,8 cm |
| Endhöhe | 0,899 (Tisch) | 0,895 (Tisch) |
| engste Öffnung bei Step | 719 | 136 |

**Die linke Hand ist der gültige Test.** Ihr Einsetz-Step 323 fällt genau in das Beugefenster, das
die npz-Gegenprobe aus Lauf 28 vorhergesagt hatte (stärkste Beugung Step 330, > 80 % ab Step 320).
Dort greift die Hand zu, hebt den Würfel 7,9 cm an und hält ihn über 202 Steps.

**Die rechte Hand hat zu früh ausgelöst.** Trigger bei Step 91, echtes Zugreifen laut npz erst bei
Step 186 (> 80 % ab Step 102). Der Würfel wurde in eine Hand gesetzt, die noch gar nicht griff.
Die 0,2 % Haltequote sind ein Artefakt der 70-%-Heuristik, **kein Physik-Befund** — die Schwelle
bezieht sich auf das bis dahin gesehene Maximum, und das ist früh in der Episode noch nicht das
globale.

**Die 24 % ehrlich gelesen.** Das liegt unter der 0,5-Schwelle der Entscheidungsregel oben — aber
die Regel war für ein binäres Ergebnis geschrieben und trifft einen dritten Fall nicht, den sie
nicht vorsah: der Griff **bildet sich und rutscht dann**. Das ist etwas anderes als „der Griff
bildet sich nie", und für RL ist es der entscheidende Unterschied.

> ⚠️ **Vorbehalt — der Test setzt den Würfel in die Fingergeometrie hinein.** Der Würfel wird auf
> den Schwerpunkt der drei Fingerspitzen gesetzt; bei einer Greiföffnung von 7,2 cm und 5 cm
> Kantenlänge überlappt er dabei die Finger. PhysX löst Durchdringung mit harten Impulsen auf.
> Damit ist offen, wieviel der 7,9 cm getragener Hub ist und wieviel Auflöse-Impuls — und die
> −1,8 cm rechts sind eher „herausgeschossen" als „fallengelassen", also kein Reibungsbefund.
> **Was für einen echten Griff spricht:** 202 Steps innerhalb 4 cm am Fingerdreieck sind rund
> 6,7 s. Ein Penetrations-Impuls ist ballistisch und in Millisekunden vorbei, nicht über
> 200 Steps. Die Zahl ist also nicht wertlos, aber kein sauberer Beweis.

#### Die Würfel „springen" im Video — das ist der Test, kein Bug

Wer sich `replay_episode0.mp4` aus einem `GRASP_MODE=hold`-Lauf ansieht, sieht Würfel ohne
Roboterkontakt von einer Position zur anderen springen. Das ist die **eingebaute Teleportation**
des Tests, nicht Physik. Das Video hat 1173 Frames bei 30 fps (39,1 s), also **exakt ein Frame je
Step** — die Ereignisse lassen sich damit direkt zuordnen:

| Ereignis | Step | Videozeit | Würfel |
|---|---|---|---|
| Einsetzen rechte Hand | 91 | **3,0 s** | `block_1` (grün) |
| Einsetzen linke Hand | 323 | **10,8 s** | `block_0` (rot) |

Beide Einsetzungen sind One-Shot (`hold_step[h] < 0`-Guard in
[`run_g1_dex3_replay.py`](../../Simulation/g1_dex3_sim/run_g1_dex3_replay.py)), und `block_2`
(gelb) wird in diesem Modus nie angefasst. Springt im Video etwas **anderes** als diese zwei
Ereignisse, ist das ein echter Befund — dann lohnt der Blick auf Durchdringungen beim Einsetzen.

**Nächster Schritt — das entscheidet das Video, nicht die nächste Messung:** Sekunde 10,8 bis
etwa 18 ansehen. Fährt der rote Würfel ruhig mit der Hand mit, trägt der Griff. Wird er
weggeschleudert und der Hub ist ein einzelner Sprung, ist es ein Impuls-Artefakt und der Test
braucht ein Einsetzen ohne Überlappung (Würfel eine Kantenlänge vor die Kuppen statt in ihren
Schwerpunkt).

Unabhängig davon ist der RL-Pfad **entblockiert**: ein Anheben ist in dieser Sim möglich, der
Shaped Reward kann also ein Signal liefern. Die verbleibende Frage ist nicht mehr „geht es
überhaupt", sondern „wie zuverlässig" — und das ist eine Frage, an der RL arbeiten kann.

#### Lauf 30: die vorregistrierte Regel ist geschlossen — die Politik greift nicht

`runs/20260812/02` (2026-08-12), Closed-Loop-Eval, 2 × 40 s, `EXECUTION_HORIZON=8`, `DR_ENABLED=0`.
Der erste Eval seit dem Einbau von `finger_span_max_rad` — damit liegt die zweite Hälfte der
Entscheidungsregel aus [Schritt 3](#schritt-3--bc-erfolgsrate-in-der-sim-server_rl_runsh-eval) vor.

| | Fingerspanne (Kommando) | Verhältnis |
|---|---|---|
| Dataset / menschliche Demo (Lauf 29) | **2,094 rad** ≈ 120° | — |
| Policy, Episode 1 / 2 | **0,347 / 0,390 rad** ≈ 22° | **19 %** |

Die Politik kommandiert rund ein Sechstel der Greifbewegung, die in der Demonstration steckt. Das
ist kein misslungener Griff, sondern keiner. Die Annäherung zeigt dasselbe Muster — ab den in
Lauf 28 korrigierten Fingerkuppen gemessen:

| Episode | Abstand Start → min | bei Step | Würfel verschoben |
|---|---|---|---|
| 1 | 14,4 → **6,3 cm** | 272 | 3,8 cm |
| 2 | 14,3 → **4,6 cm** | 984 | 1,3 cm |

Bei 5 cm Kantenlänge sind 4,6 cm zur Würfel*mitte* rund 2 cm zur Oberfläche: nah dran, aber die
Finger umschließen nichts. Beide Defizite zeigen in dieselbe Richtung — die Politik produziert eine
**zum Mittelwert gezogene, verwaschene Version** der Demonstration. Das ist die Signatur einer
Policy, die nicht erkennt, in welcher Phase der Aufgabe sie ist, und deshalb nahe der
Durchschnittsaktion ausgibt: das erwartete Verhalten bei Out-of-Distribution-Eingaben.

**Damit ist die Regel erfüllt:**

| Bedingung | Ergebnis |
|---|---|
| (a) `max_cube_lift_cm` > ~2 | ✅ 7,9 cm (Lauf 29) |
| (b) Fingerspanne klein | ✅ 19 % der Demo (Lauf 30) |
| **Konsequenz laut Regel** | **Politik/Wahrnehmung → `TUNE_VISUAL=1`** |

#### Gate vor den 48 GPU-Stunden: drei Erklärungen, eine schon widerlegt

Ein **Normalisierungs- oder Skalierungsfehler in der Aktions-Dekodierung erzeugt exakt dieselbe
Signatur** wie ein Domain-Gap — systematisch gestauchte Ausgaben über eine ganze Aktionsgruppe.
`TUNE_VISUAL` würde ihn nicht beheben. Dass ausgerechnet die Finger-Dimensionen um Faktor 5
gestaucht sind, während die Arme plausibel fahren, ist auffällig genug, um das vor einem 48-h-Lauf
auszuschließen. Drei Kandidaten:

| | Erklärung | Status |
|---|---|---|
| **(b)** | Die De-Normalisierung staucht den Ausgang | **widerlegt**, s. u. |
| **(a1)** | Domain-Gap — die Policy kann greifen, die Sim-Bilder brechen sie | **bestätigt** (Lauf 32) |
| **(a2)** | Die Policy hat den Griff nie gelernt (Training/Daten) | **ausgeschlossen** (Lauf 32) |

> **Das Gate ist inzwischen gefahren.** Die Messung unten wurde am 2026-08-13 durchgeführt und
> ergab ein Median-Verhältnis von **1,00** — Ergebnis und Auswertung:
> [Lauf 32](#lauf-32-runs2026081301-das-span-gate-ist-entschieden--a1). Der folgende Abschnitt
> beschreibt weiterhin, *wie* die Messung läuft und *warum* die Schwellen vorab feststanden.

**(b) ist erledigt, statisch, in Sekunden** — [`check_action_norm.py`](../../Simulation/scripts/check_action_norm.py)
liest `statistics.json` + `processor_config.json` aus dem Checkpoint (reines JSON, kein torch,
keine GPU) und vergleicht sie mit `meta/stats.json` des Datensatzes:

```bash
python3 Simulation/scripts/check_action_norm.py <checkpoint-dir> <dataset-dir>
```

Ergebnis auf `checkpoint-175000`: **alle vier Aktionsgruppen stimmen auf 1e-5 mit dem Datensatz
überein**, `use_percentiles=False` (also min/max), und die maximal de-normalisierbare Fingerspanne
ist **exakt 2,0944 rad** — genau der Wert aus der Demonstration. Ein Modellausgang von normiert
−1 → +1 gäbe die volle Spanne aus. Die beobachteten 0,39 rad entsprechen einer normierten Amplitude
von **0,372 von 2,0 (18,6 %)**. Die Stauchung entsteht also **im Modell selbst**, nicht beim
Dekodieren.

Zwei Nebenbefunde aus derselben Untersuchung, die später zur Falle werden können:
`override_pretraining_statistics` ist per Default `False` und `set_statistics` **überspringt** einen
bereits vorhandenen Embodiment-Tag — hier unkritisch (die Stats stimmen ja 1:1), aber bei einem
künftigen Datensatz-Wechsel unter demselben Tag würden stillschweigend die alten Statistiken
weiterverwendet. Und `clip_outliers=True` clippt nur auf [−1, 1], kann also begrenzen, nicht stauchen.

**Offen bleibt (a1) gegen (a2)** — und das ist die teuerste Frage im Projekt, weil nur bei (a1) ein
`TUNE_VISUAL`-Lauf begründet ist. Der Diskriminator ist dieselbe Policy und dieselbe Metrik wie im
Closed Loop, aber auf **echten Datensatz-Bildern** statt Sim-Renderings:
[`finger_span_openloop.py`](../../Simulation/scripts/finger_span_openloop.py). Als Bezugsgröße
läuft die Ground-Truth-Spanne derselben Episode mit, der Vergleich hängt also nicht an einer
notierten Zahl aus einem anderen Lauf.

```bash
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh span
```

> ⚠️ **Nicht direkt mit dem Host-Python aufrufen.** Auf `ikr-ki-server-01` gibt es kein numpy
> außerhalb des Containers (`ModuleNotFoundError: No module named 'numpy'`), und Module lassen sich
> dort nicht installieren. Die Aktion `span` erledigt beides: sie kopiert das Skript per
> `docker cp` in den laufenden Container (`/scripts` ist ins Image **gebacken**, ein Edit im Repo
> erreicht ihn sonst nie — dasselbe Muster wie bei `gap` und `eval`) und startet es mit dem
> **GR00T-venv** `/app/Groot-1.6/.venv/bin/python`, nicht mit dem Isaac-Python: für reine
> Policy-Inferenz wird Isaac Sim nicht gebraucht.
>
> **Der Datensatz wird bei Bedarf selbst geholt** — und zwar in drei Schritten, nicht nur einem
> Download. Genau die Reihenfolge aus [`entrypoint.sh`](../../Training/scripts/entrypoint.sh):
>
> | Schritt | Was | Warum nötig |
> |---|---|---|
> | 1 | HF-Download `unitreerobotics/G1_Dex3_BlockStacking_Dataset` (~18 GB) | der Sim-Container lädt von sich aus nur Checkpoint + USD |
> | 2 | Konvertierung LeRobot v3.0 → v2.1 | legt zusätzlich ein Backup `*_v3.0` an → **~40 GB Spitzenbedarf** |
> | 3 | `modality_4cam.json` → `meta/modality.json` | erst damit ist der Datensatz für gr00t lesbar |
>
> **Ein bloßes `huggingface-cli download` reicht nicht** — ohne Schritt 2 und 3 fehlt
> `modality.json` und der Loader scheitert. Einmalig, rund eine Stunde; ein `clean` löscht den
> Datensatz nicht (er liegt unter dem gemounteten `/data`). Abschalten mit `SPAN_AUTO_FETCH=0`,
> anderen Pfad angeben mit `SPAN_DATASET=/data/…`, andere Episoden mit `SPAN_TRAJ_IDS="0 1 2"`.
>
> **`ffmpeg` im Sim-Image (2026-08-12).** Schritt 2 schneidet die zusammenhängenden MP4s per
> Subprozess in Einzel-Episoden und scheiterte mit `FileNotFoundError: 'ffmpeg'` — das Paket war
> im Training-Image seit jeher drin, im Sim-Image nicht, weil der Sim-Pfad den Trainingsdatensatz
> nie brauchte. [`Dockerfile.vastai`](../../Simulation/Dockerfile.vastai) hat es jetzt; bis zum
> nächsten Rebuild installiert `ensure_dataset` es zur Laufzeit nach, damit kein 60-Minuten-Build
> zwischen dir und der Messung steht. Die Laufzeit-Installation überlebt `clean` nicht.
>
> **Ein Abbruch in Schritt 2 ist ungefährlich.** `convert_v3_to_v2_standalone.py` verschiebt das
> Original erst *nach* vollständiger Konvertierung ins Backup und räumt Teilstände (`*_v2.1`,
> `*_v3.0`) beim nächsten Start selbst weg. Der 18-GB-Download bleibt erhalten; ein erneutes
> `span` setzt direkt bei der Konvertierung auf.

| Vorhersage/Ground-Truth | Lesart |
|---|---|
| **≥ 70 %** | Das Modell kann greifen, die Sim-Bilder brechen es → **Domain-Gap bestätigt, `TUNE_VISUAL` begründet** |
| **≤ 35 %** | Das Modell greift auch auf echten Bildern nicht → **ViT-Lauf ginge an der Ursache vorbei**; Training, Checkpoint-Wahl und Datenaufbereitung prüfen |
| dazwischen | Teilstauchung schon auf echten Bildern — Domain-Gap ist ein Faktor, nicht der einzige; mehr Trajektorien messen |

Die Schwellen stehen im Skript, damit die Zahl nicht nachträglich gedeutet wird. Braucht den
**echten** Datensatz mit Videos (auf dem Server unter `/data/unitreerobotics/`); lokal liegt nur
`meta/`, das reicht nicht.

**Erledigt durch diese Messung:** `EXECUTION_HORIZON` kleiner zu setzen (Neuplanen statt 8 offener
Schritte) war als billiger Hebel im Gespräch. Bei 22° kommandierter Fingerbewegung greift die Hand
auch bei perfekter Ausführung nicht — das Problem sitzt in dem, was kommandiert wird, nicht in der
Ausführung.

**Wo die Stauchung *nicht* herkommt** — drei Glieder der Kette sind ausgeschlossen, bevor der
teure Verdacht geprüft wird:

| Glied | staucht? | Beleg |
|---|---|---|
| Env (Gelenkgrenzen, PD-Regler) | nein | Replay fährt 2,09 rad kommandiert → **2,10 rad erreicht** (Lauf 29) |
| Sim-Client | nein | [`client.py`](../../Simulation/g1_dex3_sim/client.py) konkateniert die Aktionsgruppen nur, keine Skalierung |
| Dimensions-Zuordnung | korrekt | `ACTION_KEYS = [left_arm, right_arm, left_dex3, right_dex3]` → Dims 14:28 sind exakt die beiden Hände |

Die Stauchung sitzt damit **vollständig upstream**: in der Policy-Ausgabe selbst oder in der
server-seitigen De-Normalisierung von GR00T. Nur dort lohnt die Suche.

#### Gestufte Meilensteine in der Closed-Loop-Eval (2026-08-12)

Bis Lauf 30 protokollierte die Eval **kein** `max_cube_lift_cm` — die Zahl existierte nur im
Replay-Pfad. Zwischen „Würfel berührt" und „gestapelt" (0/20) fehlte damit die entscheidende
Zwischenstufe, und **gegen eine Metrik, die immer 0 ist, lässt sich keine Maßnahme bewerten**: Ein
`TUNE_VISUAL`-Lauf, der mit 0/20 zurückkommt, sagt nicht, ob er in die richtige Richtung ging.

[`run_g1_dex3_sim_eval.py`](../../Simulation/g1_dex3_sim/run_g1_dex3_sim_eval.py) schreibt deshalb
jetzt `max_cube_lift_cm` je Episode plus eine Meilenstein-Leiter. Die Schwellen sind im Code als
Konstanten dokumentiert, damit sie nicht nachträglich zurechtgebogen werden:

| Stufe | Schwelle | Herkunft der Schwelle |
|---|---|---|
| `reached` | Fingerkuppe ≤ 4 cm zur Würfelmitte | derselbe Wert, mit dem der Replay „noch in der Hand" prüft |
| `grasp_attempted` | Fingerspanne ≥ 30 % der Demo (0,63 rad) | Demo-Referenz 2,094 rad aus Lauf 29 |
| `touched` | Würfel > 1 cm verschoben (nach Karenz) | klar über dem Solver-Rauschen von ~2 mm |
| `lifted` | Würfel > 2 cm angehoben | die Schwelle der vorregistrierten Regel; Lauf 29 erreichte 7,9 cm |
| `stacked` | binärer Erfolg | unverändert |

Auf die Lauf-30-Zahlen angewandt ergibt das `touched` 2/2, alles andere 0/2 — und einen Befund,
den die Leiter sichtbar macht: **`touched` ohne `reached`.** Ein Würfel bewegt sich 3,8 cm, während
die Fingerkuppen nie näher als 6,3 cm an eine Würfelmitte kommen. Angestoßen hat ihn also etwas
anderes als die sechs Kuppen — Handfläche oder Fingerglieder. Die Stufen sind deshalb bewusst
**nicht** als erzwungen monotone Kette implementiert; genau solche Abweichungen sind die
interessanten.

#### Lauf 31 (`runs/20260812/03`): die Leiter im Einsatz — und zwei neue Zahlen

Erster Eval mit der Leiter, 5 Episoden à 40 s. Erfolgsrate weiterhin 0/5 — aber das ist ab jetzt
nicht mehr die einzige Information:

| Episode | Fingerkuppe→Würfel | Fingerspanne | **Anhebung** | Verschiebung | erreichte Stufen |
|---|---|---|---|---|---|
| 1 | 14,5 → 3,8 cm | 0,43 rad | 0,22 cm | 0,3 cm | `reached` |
| 2 | 14,0 → 4,1 cm | 0,47 rad | 0,60 cm | 7,1 cm | `touched` |
| 3 | 14,4 → 4,2 cm | 0,49 rad | **1,03 cm** | 3,3 cm | `touched` |
| 4 | 14,1 → 3,7 cm | 0,39 rad | 0,47 cm | 0,8 cm | `reached` |
| 5 | 14,2 → 2,9 cm | 0,41 rad | 0,90 cm | 3,7 cm | `reached`, `touched` |

Leiter: `reached` 3/5 · `grasp_attempted` **0/5** · `touched` 3/5 · `lifted` 0/5 · `stacked` 0/5

**Die Würfel heben sich zum ersten Mal messbar.** 0,22–1,03 cm, in *jeder* Episode ungleich null.
Das liegt unter der 2-cm-Schwelle, ist also kein Greifen — aber es ist auch nicht die 0,0, die
`grasp` im Open Loop vor Lauf 29 lieferte. Vorher war diese Zahl im Closed Loop schlicht
unsichtbar; sie ist jetzt die Größe, an der sich eine Maßnahme zuerst zeigen wird.

**Der Greif-Befund ist bestätigt und hat jetzt n=7.** Die Fingerspanne liegt über fünf Episoden
zwischen 0,39 und 0,49 rad (Median 0,43 = **20,5 %** der Demonstration) — dieselbe Größenordnung
wie die 0,35/0,39 rad aus Lauf 30. Die Streuung ist gering, das ist kein Rauschen, sondern ein
stabiles, reproduzierbares Defizit.

**Die Annäherung ist besser als Lauf 30 vermuten ließ.** 2,9–4,2 cm statt 4,6/6,3 cm — die zwei
Episoden aus Lauf 30 waren nicht repräsentativ. Damit fällt allerdings auf, dass die
`reached`-Schwelle (4 cm) **mitten in der Datenverteilung** liegt: Episode 2 verfehlt sie um 1 mm,
Episode 3 um 2 mm. Der Boolean kippt dadurch fast zufällig, und `touched` ohne `reached` in
Episode 2/3 ist eher ein Schwellenartefakt als der in Lauf 30 vermutete Handflächen-Kontakt.
**Die Rohzahl (2,9–4,2 cm bei rund 10 cm Annäherung) ist hier aussagekräftiger als die Stufe** —
die Schwelle taugt zum Verfolgen einer Veränderung, nicht zur Aussage über einen einzelnen Lauf.

#### Lauf 32 (`runs/20260813/01`): das `span`-Gate ist entschieden — (a1)

Die Diskriminator-Messung aus dem Gate-Abschnitt oben, gefahren am 2026-08-13 mit
`server_rl_run.sh span` über fünf Trajektorien à 400 Schritte. Gemessen wird dieselbe Policy mit
derselben Metrik wie im Closed Loop, aber auf **echten Datensatz-Bildern**:

| Trajektorie | Spanne Vorhersage | Spanne Ground Truth | Verhältnis | MSE |
|---|---|---|---|---|
| 0 | 2,0944 rad | 2,0944 rad | 1,000 | 0,0020 |
| 1 | 2,0699 rad | 2,0767 rad | 0,997 | 0,0016 |
| 2 | 2,0944 rad | 2,0942 rad | 1,000 | 0,0019 |
| 3 | 2,0944 rad | 2,0722 rad | 1,011 | 0,0045 |
| 4 | 1,7221 rad | 2,0582 rad | 0,837 | 0,0017 |

**Median-Verhältnis: 1,00 — also 100 %.**

**Nach der oben vorregistrierten Regel (`≥ 70 %`) heißt das: Domain-Gap bestätigt, `TUNE_VISUAL`
begründet.** Der Kontrast ist deutlich: Auf echten Bildern kommandiert dieselbe Policy die volle
Greifbewegung (1,72–2,09 rad, viermal von fünf praktisch deckungsgleich mit der Demonstration);
im Sim-Rendering sind es 0,39–0,49 rad (Lauf 31, n=7). Die Fähigkeit ist im Checkpoint vorhanden
und wird von den Sim-Bildern zerstört.

Damit ist die Kandidatenliste geschlossen:

| | Erklärung | Status |
|---|---|---|
| **(b)** | Die De-Normalisierung staucht den Ausgang | widerlegt (Lauf 30, `check_action_norm.py`) |
| **(a1)** | Domain-Gap — die Policy kann greifen, die Sim-Bilder brechen sie | **bestätigt (Lauf 32)** |
| **(a2)** | Die Policy hat den Griff nie gelernt (Training/Daten) | **ausgeschlossen (Lauf 32)** |

Die niedrigen MSE-Werte (0,0016–0,0045) stützen das zusätzlich: Es geht nicht nur die grobe
Spannweite auf, die Trajektorie selbst wird nachvollzogen. Trajektorie 4 ist mit 0,837 der
schwächste Wert und bleibt trotzdem klar über der 70-%-Schwelle.

> **Einordnung der Belegkette.** `Simulation/runs/` ist gitignored — die Rohdatei
> `runs/20260813/01/finger_span_openloop.json` ist nicht versioniert. Die Zahlen oben sind aus ihr
> übernommen; wer sie nachprüfen will, fährt `span` erneut (`SPAN_TRAJ_IDS` setzt die Episoden).

**Konsequenz für die Reihenfolge:** RL bleibt nicht der nächste Schritt. Der begründete nächste
Lauf ist `TUNE_VISUAL=1` — die Regel dafür ist jetzt vollständig geschlossen, nicht mehr nur
plausibel. Erst wenn die Politik in der Sim greift, hat ein Reward-Signal einen Startpunkt.

#### Läufe 33/34 (`runs/20260814/03`, `runs/20260814/04`): der TUNE_VISUAL-Checkpoint im Closed Loop

Die Antwort auf Lauf 32. Gemessen wird der Checkpoint, den `TUNE_VISUAL=1` produziert hat:
`g1_dex3_blockstacking_vision_v2`, **checkpoint-30000** — der beste aus dem Checkpoint-Sweep, nicht
der letzte (der ist 25 % schlechter, s. [lauf3-vision-split-auswertung.md](lauf3-vision-split-auswertung.md)).
Zwei Läufe am 2026-08-14, beide mit schwarzhändigem Asset, DR an, Exec-Horizon 8:

| Lauf | Konfiguration | Rolle |
|---|---|---|
| 33 | 5 Episoden à **60 s**, dazu `span` über 5 Trajektorien | erste Sichtung |
| 34 | **10 Episoden à 40 s** | die mit Lauf 31 vergleichbare Messung |

Lauf 33 ist als Vergleich unbrauchbar, und das ist eine Lehre für sich: 60 s gegen die 40 s aus
Lauf 31, und die Fingerspanne ist ein **Maximum über die Episode** — ein längeres Fenster hebt sie
allein dadurch. Lauf 34 wiederholt im identischen Zeitfenster und ist die Zahl, die zählt.

##### Lauf 34, Episode für Episode

| Ep | Fingerspanne | % Demo | Kuppe→Würfel | Verschiebung | Anhebung | Stufen |
|---|---|---|---|---|---|---|
| 0 | 0,589 rad | 28 % | 3,1 cm | 7,9 cm | 1,00 cm | `reached`, `touched` |
| 1 | **1,442 rad** | **69 %** | 2,3 cm | 9,4 cm | **1,68 cm** | + `grasp_attempted` |
| 2 | 0,530 rad | 25 % | 4,1 cm | 4,2 cm | 1,00 cm | `touched` |
| 3 | 0,504 rad | 24 % | 2,5 cm | 3,5 cm | 0,81 cm | `reached`, `touched` |
| 4 | 0,552 rad | 26 % | 3,2 cm | 6,3 cm | 0,87 cm | `reached`, `touched` |
| 5 | 0,735 rad | 35 % | 3,2 cm | 3,8 cm | 0,93 cm | + `grasp_attempted` |
| 6 | **1,260 rad** | **60 %** | 3,3 cm | 6,2 cm | 1,17 cm | + `grasp_attempted` |
| 7 | 0,543 rad | 26 % | 2,8 cm | 6,3 cm | 1,24 cm | `reached`, `touched` |
| 8 | 0,565 rad | 27 % | 3,1 cm | 4,1 cm | 1,02 cm | `reached`, `touched` |
| 9 | **1,673 rad** | **80 %** | 2,5 cm | 2,8 cm | 1,33 cm | + `grasp_attempted` |

Leiter: `reached` 9/10 · `grasp_attempted` **4/10** · `touched` 10/10 · `lifted` **0/10** ·
`stacked` 0/10.

##### Der Vergleich mit Lauf 31 (gleiches Zeitfenster, alter Checkpoint)

| | Lauf 31 (alt, 40 s, n=5) | **Lauf 34 (vision_v2@30000, 40 s, n=10)** |
|---|---|---|
| Fingerspanne Median | 0,43 rad = **20,5 %** | **0,577 rad = 27,6 %** |
| Bereich | 0,39–0,49 rad | **0,50–1,67 rad** |
| Episoden ≥ 60 % der Demo | 0/5 | **3/10** |
| `grasp_attempted` | 0/5 | 4/10 |
| `touched` | 3/5 | 10/10 |
| Anhebung max | 1,03 cm | 1,68 cm |
| Kuppe→Würfel | 2,9–4,2 cm | 2,3–4,1 cm |
| `lifted` / `stacked` | 0/5 | **0/10** |

**Die Trennung ist vollständig:** jede der zehn neuen Episoden liegt über jeder der fünf alten
(U = 50 von 50). Exakter einseitiger Rangtest über alle 3003 Aufteilungen: **p = 3,3 · 10⁻⁴**. Bei
diesen Stichprobengrößen ist das der bestmögliche Wert — mehr Trennschärfe gibt die Stichprobe
nicht her.

Wichtiger als der Median ist die **Form** der Änderung. Der alte Checkpoint lag über sieben
Episoden eng zwischen 0,35 und 0,49 rad — oben zu Recht als „stabiles, reproduzierbares Defizit"
beschrieben. Lauf 34 hat sechs Episoden bei 0,50–0,59, und drei brechen aus: 1,26 · 1,44 · 1,67 rad,
also 60–80 % der Demonstration. Es ist kein gleichmäßiges Anheben des Niveaus, sondern **ein neuer
Modus, den der alte Checkpoint nie gezeigt hat.**

Und dieser Modus wirkt mechanisch:

| Rangkorrelation (Spearman, n=10) | Wert | Lesart |
|---|---|---|
| Fingerspanne ~ **Anhebung** | **+0,62** | wo die Politik greift, hebt sich der Würfel am meisten |
| Fingerspanne ~ Verschiebung | +0,05 | das Wegschieben hängt **nicht** am Griff — das ist Anstoßen |
| Fingerspanne ~ Annäherung | −0,31 | schwach; näher dran heißt eher mehr Griff |

Dass die Anhebung mit der Spanne läuft, die Verschiebung aber nicht, ist genau die Signatur, die
ein wirksamer Greifbefehl haben müsste: mehr Hub ohne mehr Herumschieben. Wäre beides korreliert,
wäre die Anhebung bloß ein Nebenprodukt von mehr Kontakt überhaupt.

##### Was sich nicht bewegt hat

- **Die Annäherung.** 2,3–4,1 cm gegen 2,9–4,2 cm — unverändert. Dass `reached` von 3/5 auf 9/10
  springt, ist wieder das oben beschriebene Schwellenartefakt: Episode 2 verfehlt die 4-cm-Grenze
  um 1 mm. Die Rohzahl zählt, nicht die Stufe.
- **`lifted` bleibt 0/10.** Beste Anhebung 1,68 cm gegen die 2-cm-Schwelle. Näher als je zuvor,
  aber darüber ist es nicht.
- **Die Lücke zum Realbild bleibt groß.** 27,6 % in der Sim gegen 100 % auf echten Bildern —
  Faktor 3,6. Das `span`-Gate aus Lauf 33 (derselbe neue Checkpoint, fünf Trajektorien) liefert
  erneut Median 1,00: die Realbild-Fähigkeit hat das Vision-Training unbeschadet überstanden.
  Die MSE liegt dabei mit 0,0025–0,0062 **über** den 0,0016–0,0045 aus Lauf 32 — nicht
  überinterpretieren, Lauf 1 hat diese fünf Episoden 175 k Schritte lang gesehen, das ist
  Memorisierung und kein fairer Vergleich.

##### Zuordnung: es war der Checkpoint

Zwischen Lauf 31 (2026-08-12) und Lauf 34 hat sich auf der Sim-Seite nichts geändert, was die
Aktionen berührt. `83d6ded` fasst nur `policy_latency.py` an (Diagnose-Bench, nicht im Eval-Pfad),
`7a05c9d` führt `SCENE_CAM` und `CAM_RES_SCALE` ein — **beide standardmäßig aus** und in beiden
Läufen nicht gesetzt. `557fb3b` ist reine Zeitmessung. Einzige geänderte Variable ist der
Checkpoint.

##### Konsequenz

Die vorregistrierte Frage lautete: *„Bewegt sie sich nicht deutlich über 19 %, ist der Vision-Pfad
ausgereizt."* Sie hat sich bewegt — signifikant und mit einem qualitativ neuen Verhalten.
**Der Vision-Pfad ist also nicht ausgereizt, schließt den Gap aber auch nicht.** Rund 48
GPU-Stunden haben ungefähr ein Drittel des Weges gebracht; die restlichen zwei Drittel kommen
nicht durch mehr vom Gleichen.

Damit wird **Co-Training auf gerenderten Bildern** (Schritt 4 in [next-steps.md](../../next-steps.md))
zur begründeten Eskalation: der Encoder muss domäneninvariant werden, nicht nur besser auf
Realbildern. Die +0,62-Korrelation ist das Argument, dass sich das auszahlt — steigt die Spanne
weiter, folgt die Anhebung mit, und bis zur 2-cm-Schwelle fehlen noch 0,3 cm.

RL bleibt hinten an: die BC-Baseline steht weiterhin bei 0 Erfolg, das Reward-Signal hätte
weiterhin keinen Startpunkt.

> **Einordnung der Belegkette.** `Simulation/runs/` ist gitignored — `runs/20260814/03` und `04`
> (jeweils `results.json` plus Log) sind nicht versioniert. Die Zahlen oben stammen aus ihnen.
> Nachfahren lässt sich das mit
> `NUM_EPISODES=10 EPISODE_LENGTH_S=40 CHECKPOINT_PATH=/data/checkpoints/groot-g1dex3-vision-v2-30000 ./Simulation/server_rl_run.sh eval`.
> Seit dem 2026-08-14 schreibt die Eval den ausgewerteten Checkpoint als Feld `checkpoint`
> (Pfad, `run_id`, `global_step`, Gewichtsgrößen) in `results.json` — bei diesen beiden Läufen
> fehlt es noch, ihre Identität ist über die Logzeile `Checkpoint: …` belegt.



---

### Läufe 35–37 (`runs/20260822/01`–`03`): Die Würfellage kommt aus dem Bild, nicht aus der Hand

Vollständige Auswertung mit allen Zahlen:
[wuerfellage-rekonstruktion-lauf35-befund.md](../simulation/wuerfellage-rekonstruktion-lauf35-befund.md).
Hier nur, was für spätere Läufe zählt.

**Lauf 35** — erster Abnahmelauf von `replay-calibrate` (v4, 40 Episoden): 10 Anker, alle Gates
gefallen. Ursachen: `track_colors` mittelte über die ganze Farbmaske statt über den größten Blob
(Gelb löste im Median bei Frame 18 einen Bewegungsbeginn aus, bevor der Roboter den Würfel
berührt), und der Anker lag 4–11 cm neben dem Würfel.

**Lauf 36** — nach der Blob-Korrektur: die Zeitmessung ist repariert (Grün-Onset-Median 216 → 588,
Kameradifferenz 50 → 2 Frames), die Ankerausbeute fällt trotzdem auf 3. Der Grund beendet das
Verfahren: **bei allen 60 erkannten Schließintervallen ist die Hand am Ende noch rund 10 cm offen**
(Median 0,108 m), bei 5 cm Würfelkante. Was als Griff detektiert wird, ist ein Zucken einer weit
offenen Hand. Das deckt sich mit dem `scan`-Lauf vom 2026-08-17, wo bei 101 von 116 Griffen die
engste Kuppenöffnung über 6 cm lag. **Die Fingeröffnung taugt für diese Hand nicht als
Greifdetektor** — weder im alten Pfad noch in v4.

**Lauf 37** — `cams` + `layoutcheck` gegen gerenderte Bilder mit bekannter Grundwahrheit. Das
Pinhole-Kameramodell trifft die Würfellage auf **0,29 cm** (`cam_left_high`) und **0,40 cm**
(`cam_right_high`). Eine Suche über Brennweite (45–100°), Blickziel und Kamerahöhe findet keine
bessere Anpassung — das Modell braucht keine Korrektur.

#### Zwei Konsequenzen für alle folgenden Läufe

1. **`block_z_surface` ist von 0,915 auf 0,895 korrigiert** (Commit `564b2ea`). Der Tisch ist 0,87 m
   hoch, ein 5-cm-Würfel ruht also auf 0,895. Die 0,915 waren eine bewusste Erhöhung, um die
   Würfeloberkante ohne Tischanhebung auf 0,94 zu bringen — sie erreichte ihr Ziel nie, weil die
   Würfel die zwei Zentimeter fielen und doch auf 0,895 landeten. **Die Ruhelage ändert sich
   nicht**, nur der Fall beim Reset entfällt. Erfolgskriterien rechnen mit Höhendifferenzen und
   verschieben sich nicht. Wer Läufe vor und nach `564b2ea` vergleicht, sollte die ersten
   Physikschritte nach dem Reset im Blick behalten.
2. **Die falsche Ebene kostete rund 1 cm** in jeder Rückprojektion Bild → Tisch: Restfehler 1,11 cm
   statt 0,29 cm, mit einem systematischen Versatz von −1 cm in x. Ältere Zahlen aus
   `extract_block_layout` sind entsprechend zu lesen.

Der Blickzielpunkt in `camera_geometry.py` behält seine 0,915. Das ist ein Kameraparameter, keine
Würfelhöhe — genau diese Pose ist geprüft, und sie darf bei einer Ebenenkorrektur nicht mitwandern.

---

### Läufe 38–46 (`runs/20260823/02`, `runs/20260824/05`–`12`): Die Fingergrundgelenke standen still

**Lauf 38** (`render`, `runs/20260823/02`) lieferte erstmals einen vollständigen Co-Training-Datensatz —
und einen Widerspruch, der sich nicht mehr auf die Würfellage schieben ließ: über alle Frames und beide
Hände kam **keine Fingerkuppe je näher als 10 cm** an den Würfel, den das Bild-Layout gesetzt hatte.
Der reale Würfel bewegt sich am Fensterende nachweislich, die reale Hand hält ihn also. Entweder war die
Würfellage falsch — oder die Handposition aus dem aufgezeichneten Zustand.

**Läufe 39–40** (`tipcheck`, `runs/20260824/05`, `/06`) trennen das mit
[`project_fingertips_check.py`](../../Simulation/g1_dex3_sim/project_fingertips_check.py): der Roboter
wird auf den aufgezeichneten Zustand gesetzt, dann werden Fingerkuppen, Kette Becken→Handgelenk und
Layout-Würfel ins **reale** Bild desselben Frames projiziert. Lauf 40 ergänzte eine Vorzeichen-Probe über
sieben Spiegelungsvarianten. Ergebnis am Greifframe dreier Episoden, bei 5 cm Würfelkante:

| Variante | rechte Kuppenöffnung | Abstand zum Würfel |
|---|---|---|
| damals aktiv (`[17, 19, 24, 26]`) | 8,0 / 12,2 / 10,8 cm | 13,5 / 17,5 / 12,3 cm |
| ohne rechte Spiegelung | 5,1 / 7,3 / 8,7 cm | 10,4 / 11,5 / 9,4 cm |

**Lauf 41** (`runs/20260824/07`) misst zusätzlich je Handdimension die aufgezeichnete Spannweite gegen die
Gelenkgrenze — und kippt damit auch die verbliebene linke Spiegelung. Die `_1`-Beugegelenke sind der
Kronzeuge: sie wurden **nie** gespiegelt und passen trotzdem beide exakt in ihre Grenzen.

| Gelenk | Datensatz (ganzer Korpus) | Grenze |
|---|---|---|
| `left_hand_index_1` | −2,083 … −0,008 | −2,13 … 0,05 |
| `right_hand_index_1` | 0,010 … 2,085 | −0,05 … 2,14 |
| `left_hand_index_0` | −1,089 … 0,267 | −1,571 … 0,0 |
| `right_hand_index_0` | −0,199 … 1,646 | 0,0 … 1,571 |

Der Datensatz ist also bereits **seitenweise in USD-Konvention** aufgezeichnet: links negativ =
schließen, rechts positiv = schließen. Die Annahme „Datensatz: positiv = schließen", auf der der
Sign-Convention-Fix von §14.2 der [Umsetzungsnotizen](../simulation/umsetzungsnotizen.md) beruhte, ist
falsch. Gespiegelt wurden die Werte damit aus ihrer Grenze heraus, wo `set_joint_position_target` sie auf
0 klemmt: **das Gelenk bewegte sich überhaupt nicht.** In Episode 0 klemmten auf den linken `_0`-Gelenken
550 bzw. 587 von 1173 Frames.

#### Tragweite

`_SIGN_FLIP_IDX` wirkt in `_get_observations` **und** `_pre_physics_step` — die Politik sah verdrehte
Fingerwinkel und ihre Aktionen wurden aus der Grenze gedreht. Das hebt sich nicht auf. Betroffen ist
damit **jeder bisherige Lauf**: Eval, `grasp`, RL, Replay, Co-Training. Alle Greifzahlen dieser Chronik
sind mit vier stillstehenden Fingergrundgelenken entstanden.

Das erklärt die Kette rückwärts: Hand schließt nie → bei 101 von 116 Griffen bleibt die engste
Kuppenöffnung über 6 cm (`scan`, 2026-08-17) → die v4-Greifanker sind unbrauchbar (Lauf 36) → der darauf
gebaute `close_step`-Detektor misst Rauschen. Auch Lauf 26 („Griff scheitert auch ganz ohne Modell",
`max_cube_lift_cm` 0,0) und Lauf 30 („Politik kommandiert nur 19 % der Demo-Fingerspanne") sind unter
diesem Vorbehalt neu zu lesen.

#### Was geändert wurde

- `_SIGN_FLIP_IDX = []` — keine Spiegelung, auf keiner Seite.
- `DATASET_INIT_STATE`: alle 28 Startwerte roh aus dem Datensatz, keine Umrechnung mehr.
- `_widen_finger_joint_limits`: die vier `_0`-Gelenke sind wieder aufgenommen. Beide Hände fahren real
  ~0,2 rad über die Nulllinie in die Gegenrichtung, rechts `index_0` zusätzlich über 1,571 hinaus — die
  USD-Grenze endet dort auf beiden Seiten exakt an der Null. Neue Grenzen = Union(USD, Datensatz-Min/Max)
  + ~0,05 rad, wie bei den `_1`-Gelenken. Maßgeblich ist `observation.state` (was das Gelenk erreicht
  **hat**), nicht `action`: kommandiert wurde stellenweise deutlich mehr (`left_index_0` bis −1,762 gegen
  −1,089 erreicht), das hat auch die reale Hand nicht ausgefahren.
- Die Weitung kommt zu spät für den Spawn (Lauf 42, `runs/20260824/08`): Isaac Lab prüft `init_state`
  gegen die **Original**-USD-Grenzen und wirft dort einen `ValueError` — noch in `super().__init__()`,
  lange bevor `_widen_finger_joint_limits` laufen kann. Es klemmt also nicht still, es bricht ab.
  Deshalb spawnt der Roboter jetzt mit auf die Grenze gekappten Werten (`SPAWN_JOINT_POS`, rein aus
  `DATASET_INIT_STATE` abgeleitet); die echten Werte schreibt `_widen_finger_joint_limits` direkt danach
  in `default_joint_pos`, aus dem `_reset_idx` liest. Der gekappte Zustand lebt bis zum ersten Reset.

#### Was damit nicht erklärt ist

Rund **10 cm** Abstand bleiben zwischen der jetzt schließenden Hand und der Würfellage aus dem Bild; die
Vorzeichenkorrektur nimmt etwa 3 cm davon. Diesen Rest zerlegen die Läufe 43–44.

Der Messpunkt war zunächst der Bewegungsbeginn — der eine Frame, an dem der Würfel anfängt, sich zu
bewegen. `layout.json` nennt aber seine **Ruhelage**: bis der Detektor anschlägt, hat die Hand ihn längst
transportiert, und der Versatz misst den Transport statt eines Fehlers. `tipcheck` fährt deshalb seit
Lauf 44 ein **Anflugprofil** über die ganze Episode (`--approach-stride`, kein Renderdurchgang, nur ein
`sim.forward()` je Frame) und meldet je Hand den nächsten Punkt mit Frame, Würfel und Versatzvektor.

Der erste Anlauf (Lauf 45, `runs/20260824/12`) suchte das Minimum über die **ganze** Episode und ist
damit hinfällig: seine Minima lagen samt und sonders NACH dem Bewegungsbeginn, bis f816 von rund 1000
Frames. Dort trägt die Hand den Würfel längst, verglichen wird aber gegen seine Ruhelage — die Messung
erfasste genau den Transport, den sie ausschließen sollte. Ein daraus gezogener Zwischenschluss („die
Kamerapose ist entlastet, übrig bleibt ein reiner Höhenversatz von 7 cm") trägt nicht. Das Anflugprofil
sucht deshalb seit Lauf 46 nur im Fenster **vor** dem Onset, wo die Ruhelage gilt.

Zwei Befunde aus Lauf 45 bleiben gültig, weil sie nicht vom Würfelort abhängen:

1. **Alle 48 Kuppenwerte liegen über der Würfelmitte**, Minimum +0,7 cm, im Mittel +4 bis +9. Kein
   einziges Umschließen — die Hand steht in jedem gemessenen Moment über dem Würfel, nie um ihn herum.
2. **Der Messkörper ist entlastet.** Die Kuppenspreizung beträgt 5,1 cm bei 5 cm Würfelkante. Säßen die
   Messpunkte am Handgelenk statt auf den Kuppen — die Ursache von Lauf 28 —, lägen sie dichter
   beieinander. Die Hand ist sauber geschlossen; sie sitzt nur zu hoch.

Offen bleibt damit, wie groß der Höhenversatz im gültigen Fenster wirklich ist und wo er herkommt.
Kandidaten: die Basishöhe `pos=(0.0, 0.0, 0.85)` gegen die Tischhöhe 0,87, und die im 28-dim-State
fehlenden Hüft-/Waist-Gelenke — Letztere passen schlecht zu einem rein vertikalen Versatz, weil eine
Rumpfneigung vorwiegend waagerecht verschiebt (13° ≈ 13 cm nach vorn, aber nur ~1,5 cm nach unten).

Die früher notierte Einschätzung „die Handgelenk-Marken sitzen auf den realen Handgelenken, also stimmt
die Kamerapose" war Augenmaß an einem Bildausschnitt und trägt ebenfalls nichts.

#### Lauf 47 (`runs/20260824/13`): die erste belastbare Messung des Versatzes

Mit der Fenstergrenze vor dem Onset, handelnde Hand, acht Episoden:

| | Mittel | Streuung | Spanne |
|---|---|---|---|
| dx | +6,1 | 4,0 | −2,5 … +10,4 |
| dy | **−0,2** | 3,9 | zentriert auf null |
| dz | **+8,6** | 2,1 | +5,9 … +13,3 |

**Sechs der acht Minima liegen direkt an der Fenstergrenze** — die Hand nähert sich monoton bis zum
Bewegungsbeginn und wird dort abgeschnitten. Näher kommt sie bis dahin nicht, und ab dem Onset bewegt
sich der Würfel, die Hand muss ihn also halten. Der Versatz ist echt, kein Zeitartefakt. Er liegt bei
dy = 0 sauber in der x-z-Ebene.

Zur **Rumpfneigung** ist eine frühere Rechnung dieser Chronik zu korrigieren: „13° ≈ 13 cm nach vorn,
nur 1,5 cm nach unten" rechnete mit dem Hebel Waist→Schulter. Maßgeblich ist Waist→**Hand**, und die
steht beim Greifen fast waagerecht nach vorn ab (r ≈ 0,34 m). Ein Pitch dreht sie damit vorwiegend
**vertikal**: 8,6 cm entsprächen 14,5° Vorneigung bei nur ~1 cm horizontal. Rumpfneigung war also
verfrüht ausgeschlossen.

Getrennt werden Neigung und Basishöhe über die Signatur: ein Pitch skaliert dz mit der Handreichweite,
eine falsche Basishöhe nicht. Gemessen: **r = −0,27** zwischen dz und Hand-x — kein positiver
Zusammenhang, also kein Pitch-Signal, sondern ein konstanter Versatz. Bei n = 8 und nur 14 cm
Reichweitenspreizung ist das ein Indiz, keine Entscheidung.

Für den Ein-Parameter-Test ist die Beckenhöhe jetzt über `ROBOT_BASE_Z` einstellbar (Default 0,85, wie
bisher). Die Kamerapose wandert dabei ausdrücklich **nicht** mit: sie steht statisch in
`camera_geometry.py` und ist gegen Grundwahrheit auf 0,3 cm geprüft. Sinkt dz bei `ROBOT_BASE_Z=0.764`
gegen null, ohne dx zu verschlechtern, ist die Ursache gefunden.

#### Lauf 48 (`runs/20260824/15`): die Beckenhöhe war es

`ROBOT_BASE_Z=0.764` gegen die Referenz 0,85, gleiche acht Episoden:

| | 0,85 (Lauf 47) | 0,764 (Lauf 48) |
|---|---|---|
| dx | +6,1 | +5,7 |
| dy | −0,2 | −0,8 |
| **dz** | **+8,6** | **+1,0** |
| Betrag | 11,9 | 7,6 |

Der Höhenversatz verschwindet **1:1 mit der eingestellten Absenkung**, dx und dy bleiben unberührt —
genau die Signatur eines starren Versatzes. Und erst jetzt **umschließen die Fingerkuppen den Würfel**:
rechte Hand −0,9/−4,0/−3,3 bzw. +1,3/+0,0/−2,0, also gemischte Vorzeichen im Bereich ±3 cm um die
Würfelmitte, bei 5 cm Kante mithin im Würfel. In Lauf 47 lagen alle 48 Messwerte darüber.

Der Wert ist auch unabhängig plausibel: die Beckenhöhe des stehenden G1 liegt bei ~0,76 m. Die 0,85
waren ein Ansatz ohne Quelle. `pos=(0.0, 0.0, 0.764)` ist jetzt der Default, `ROBOT_BASE_Z` stellt den
alten Wert wieder her.

**Tragweite:** die Reichweite zum Tisch ändert sich in **jedem** Lauf — Eval, Grasp, RL, Replay,
Co-Training. Greifzahlen von vor dieser Korrektur sind nicht direkt vergleichbar. Zusammen mit
`_SIGN_FLIP_IDX` (Läufe 38–44) sind das zwei Geometriefehler, die jede bisherige Greifmessung dieser
Chronik betreffen.

**Offen bleibt dx = +5,7 cm** (Streuung 3,8) bei dy = 0 — ein Versatz nach vorn, deutlich kleiner als
der behobene und mit größerer relativer Streuung. Nächster Kandidat, jetzt ohne den vertikalen Anteil,
der ihn bisher überdeckte.

#### Läufe 49–51: auch die Basis in x — die Rekonstruktion steht

Der erste x-Test lief mit falschem Vorzeichen (`+0.057`) und trieb dx von +5,7 auf +9,6. Die Richtung
ergibt sich aus derselben Regel wie bei der Höhe: dx = Hand − Würfel = +5,7 heißt, der Roboter steht zu
weit **vorn**, muss also zurück. Mit `ROBOT_BASE_X=-0.057`:

| | (0, 0,85) Ausgang | (0, 0,764) | (−0,057, 0,764) |
|---|---|---|---|
| dx | +6,1 ± 4,0 | +5,7 ± 3,8 | **+1,5 ± 2,2** |
| dy | −0,2 ± 3,9 | −0,8 ± 3,7 | **−0,1 ± 3,4** |
| dz | +8,6 ± 2,1 | +1,0 ± 2,0 | **+0,9 ± 2,0** |
| Betrag | 11,9 | 7,6 | **4,5 ± 1,7** |

Getragen wird die x-Korrektur nicht vom Mittelwert, sondern von der **Streuung**: die fällt auf jeder
Achse. Eine bloße Verschiebung täte das nicht — sie verschöbe den Mittelwert und ließe die Streuung
stehen. Bei 5 cm Würfelkante bedeutet |d| = 4,5 cm, dass die Hand am Würfel ist.

Beim x-Test ist allerdings der Anflugpunkt kein stabiler Vergleichspunkt: im Fehlversuch mit `+0.057`
wanderten dy und dz mit (−0,8 → +2,9 bzw. +1,0 → +3,5), obwohl nur x verschoben wurde — das Minimum
sprang auf andere Frames. Beim z-Test war der Effekt groß genug, um dasselbe Minimum zu halten (dz fiel
exakt 1:1, dx und dy blieben unberührt). Für kleine Verschiebungen ist die Methode also unscharf.

**Damit ist die Würfellage-Rekonstruktion belastbar.** `pos=(-0.057, 0.0, 0.764)` ist der neue Default,
`ROBOT_BASE_X` / `ROBOT_BASE_Z` stellen die alten Werte wieder her. Der Weg dahin führte über drei
unabhängige Geometriefehler — die Vorzeichenspiegelung der Fingergrundgelenke, die Beckenhöhe und die
Basis in x —, die alle zusammen die rund 12 cm ergaben, an denen das Verfahren zuvor scheiterte.

#### Lauf 52: der Gierwinkel fehlte in jedem Lauf

Die Würfel liegen im Realdatensatz teils schräg zur Tischkante. Die Sim stellte sie ausnahmslos
achsparallel: `place_cubes` schrieb ein festes Identitäts-Quaternion, `_reset_idx` ebenso. Das
betrifft nicht nur den Co-Training-Render, sondern **jeden** Sim-Lauf — Eval, Replay, RL.

Für das Co-Training ist das derselbe Fehler wie eine falsche Position, nur im Drehfreiheitsgrad: das
Paar ist (Sim-Bild, **Real**-Aktion), und die Realaktion richtete die Hand nach einem Würfel bei ψ
aus, während das Bild ihn bei 0° zeigt. Für den Griff kommt hinzu, dass ein 5-cm-Würfel über die
Fläche 5,0 cm misst und über die Diagonale 7,1 cm — bei falscher Drehung trifft die Hand eine Ecke.
Die Größenordnung deckelt sich allerdings selbst: mehr als ±45° kann der Fehler wegen der
4-Zähligkeit nicht sein, das sind rund 1 cm je Kontaktseite. Die 6,8 cm Restabstand aus Lauf 20
erklärt das **nicht** allein.

**Zurückgezogen — die Begründung dieser Änderung stützte sich auf eine ungültige Messung.** Ich
hatte den Vorgänger `top_face_blob` (Min-Area-Box im Pixelraum, `axis_uv`) auf die PNGs in
`runs/20260822/01/calibration_report/` losgelassen und daraus abgeleitet, er raste in 101 von 120
Fällen auf exakt 0,0° ein, die Farbmaske sei nur zu 44 % gefüllt und pro Farbe 99,5 / 39,8 / 58,3 %.

Diese Dateien sind **annotierte Debug-Ausgaben, keine Datensatz-Frames**. Nachweis: 336 Pixel exakt
(255,255,255) als geschlossener Rechteckrahmen und 76 Pixel exakt (255,0,0) — reines gesättigtes Rot
kommt in einem Foto nicht vor. Der eingezeichnete Farbring ist damit der hellste gesättigte Bereich
des Blobs, und der Deckflächenschnitt (hellste 30 %) griff **den Marker statt der Würfeloberseite**;
der weiße Rahmen zerschneidet zusätzlich die Farbmaske. Alle Zahlen aus dieser Quelle sind hinfällig,
einschließlich der beiden Nachbesserungsversuche daran (Otsu-Schwelle, Löcher füllen).

Ungültig heißt hier nicht „das Gegenteil stimmt": ob `axis_uv` auf echten Frames einrastet, ist
damit schlicht **ungemessen**. Das Argument bleibt geometrisch plausibel — eine Min-Area-Box über
eine binarisierte Kleinmaske bevorzugt die Bildachsen, und sie misst die perspektivische Verzerrung
mit —, aber es hat keinen Beleg mehr hinter sich.

**Was die Messung überlebt:** die synthetische Abnahme rendert ihre eigenen Bilder und ist unberührt.
Und der Probelauf auf dem Server (Episoden 0, 1, 2, 8, 12) ging über die echten Videos: **1 von 15
Würfeln durch das Tor (7 %)**, dieser eine bei **30,0°** Schräglage. Der Ertrag ist damit belegt, die
*Ursache* dafür nicht — sie war es nie.

Das Verfahren selbst: Deckflächenpixel **erst zurückprojizieren** (auf der eigenen Ebene
ist die Fläche wieder ein Quadrat), dann das **4. Winkelmoment** — die Harmonische, die zur
4-Zähligkeit gehört. Kein Raster zum Einrasten. Dazu zwei Tore: die **Formprobe**
(Diagonale/Kante, ideal √2) und die **Einigkeit beider Kameras**.

| synthetisch, alle Rauschstufen | Fehler Median | p90 | Ausreißer > 20° |
|---|---|---|---|
| ohne Tore | 3,15° | 26,3° | — |
| mit Formprobe ≥ 1,25 und Einigkeit ≤ 8° | **0,12°** | **1,68°** | 1 von 160 |

Zwei Zwischenschritte waren Fehlgriffe und stehen hier, damit sie nicht wiederholt werden. Die
Breite als **Auswahlkriterium** statt als Prüfmerkmal zu nehmen machte es schlechter (Ausreißerquote
0 % → 5,2 % bei σ=0,01) — der 45°-Umschlag entsteht, wenn die Punktwolke gar kein Quadrat mehr ist,
und dann rettet keine Auswahlregel. Und die Breite als **Perzentilspanne** (5.–95.) zu messen tötet
die Unterscheidung: quer zur Diagonale ist die Projektion dreieckig verteilt, quer zur Kante
gleichverteilt, das Verhältnis fällt von 1,41 auf 1,08. Es muss die volle Spannweite sein.

**Auf echten Frames passiert 1 von 15 Würfeln das Tor (7 %, Probelauf über fünf Episoden).** Warum
so wenige, ist offen. `locate_cubes` legt die Diagnosefelder je Kamera in `layout.json` ab
(`fill`, `top_px`, `top_width_cm`, `top_squareness`, `yaw_coherence`); die Antwort steht dort und
muss aus dem Probelauf gelesen werden, nicht aus Debug-PNGs.

Bis dahin bleibt `cubes_yaw_deg` für die übrigen Würfel `null`, und der Renderer verhält sich für sie
wie bisher. `null` heißt „nicht belastbar gemessen", nicht „liegt gerade" — der Unterschied steht im
Code, weil er sonst beim nächsten Lesen verlorengeht.

Neu zur Prüfung: `server_rl_run.sh yawcheck` (synthetische Abnahme, ohne Isaac und ohne Datensatz)
und `LAYOUTCHECK_EXPECT_YAW` für die Abnahme gegen den echten Renderer. Details in
[wuerfellage-rekonstruktion.md §7](../simulation/wuerfellage-rekonstruktion.md#7-gierwinkel-um-die-eigene-z-achse).
