# LaTeX — Projektarbeit (Ausarbeitung)

> [!NOTE]
> **Stand: 2026-10-11** (28 Seiten Hauptteil / 48 Seiten gesamt, baut sauber durch). Umbau nach
> Absprache mit dem Betreuer: kein Theorie-/Stand-der-Technik-Kapitel, Kap. 3 geht jede
> Teilaufgabe (Docker, Training, Simulation, Validierung, Entwicklungswerkzeuge) einzeln durch und
> verweist für Details auf `docs/`. Neu: Lernkurven aller drei Läufe aus W&B (Kap. 4.1),
> Verhalten des zuletzt getesteten Checkpoints aus den Videos vom 14.08.2026 (Kap. 4.11),
> Hindernisse, Arbeitsmittel (KISSKI vs. Institutsserver) und KI-gestützte Entwicklungsumgebung
> (Kap. 7). Korrigiert: SIMPLER-Zahlen, Hyperparameter (LR 1e-4, Warmup 0,10 laut W&B),
> Isaac-Sim-Version (5.1 statt „4.x“), Einschränkungen des Fingeröffnungstests (Trainingsepisoden),
> Statistik (Konfidenzintervalle, t-Test-Details), Bib-Eintrag πRL. **Noch offen:**
> Matrikelnummern A/B und Abgabedatum in `preamble/settings.tex`.

## Zweck

Schriftliche Ausarbeitung der Projektarbeit (Fine-Tuning von NVIDIA GR00T N1.6 auf dem
Unitree G1 + DEX3-Hand). Quelle des Inhalts ist die Projektdokumentation unter `docs/`;
diese Ausarbeitung fasst sie nicht zusammen, sondern erzählt den Arbeits- und
Erkenntnisprozess (inkl. des Domain-Gap-Befunds) und referenziert `docs/` an den
relevanten Stellen.

## Struktur

```
latex/
├── main.tex                 # Hauptdatei (scrreprt, KOMA-Script, lualatex)
├── preamble/
│   ├── packages.tex         # Pakete
│   └── settings.tex         # Einstellungen (Satzspiegel, Acronyms, minted, …)
├── frontmatter/
│   ├── titelseite.tex       # Titelseite        ← persönliche Daten offen
│   └── zusammenfassung.tex  # Zusammenfassung + englisches Abstract
├── chapters/
│   ├── 01_einleitung.tex                #  Motivation, Ziele, Beitragsübersicht/Eigenleistung
│   ├── 02_grundlagen.tex                #  Begriffe und Plattform (bewusst knapp)
│   ├── 03_methoden.tex                  #  Umsetzung je Teilaufgabe: Docker, Training, Sim, Validierung, Werkzeuge
│   ├── 04_ergebnisse.tex                #  Lernkurven (W&B), Läufe 1–3, Diagnosekette, Lauf-34-Verhalten, Geometrie
│   ├── 05_diskussion.tex                #  Einordnung, Messartefakte, Limitierungen
│   ├── 06_weiterfuehrende_arbeiten.tex  #  Co-Training (inkl. Synth-Datensatz), Cosmos, RL, N1.7, Lokomotion
│   ├── 07_projektverlauf.tex            #  Hindernisse, Arbeitsmittel, Entwicklungsumgebung/KI, Lessons Learned
│   └── 08_fazit.tex                     #  Fazit (Antworten auf F1–F3) und Ausblick
├── backmatter/
│   ├── anhang.tex           # Datensatz, SLURM, Software, Trainingsdynamik, Szene
│   └── erklaerung.tex       # Ehrenwörtliche Erklärung ← prüfen
├── figures/                 # Abbildungen
├── references.bib           # Literatur
├── Makefile / .latexmkrc    # Build-Konfiguration
└── main.pdf                 # zuletzt kompiliertes PDF (gitignored)
```

## Kompilieren

LuaLaTeX + Biber, `-shell-escape` für `minted`. Am einfachsten via `latexmk` (Konfiguration
liegt in `.latexmkrc`):

```bash
latexmk main.tex      # oder einfach:  make
```

Manuell:
```bash
lualatex -shell-escape main.tex
biber main
lualatex -shell-escape main.tex
lualatex -shell-escape main.tex
```

Hilfs- und Cache-Dateien (`*.aux`, `_minted/`, …) sowie das PDF sind über `.gitignore`
ausgeschlossen.

## Nächste Schritte (TODO)

- [x] Inhalt aller `chapters/*.tex` ausgearbeitet (01 Einleitung … 08 Fazit)
- [x] Kompaktierung auf Zielumfang 25–35 Seiten Hauptteil (2026-08-26: 35 S. bei 11pt)
- [ ] Vollständige Namen der Teammitglieder in der Beitragstabelle (Kap. 1) eintragen
- [x] Zusammenfassung / Abstract geschrieben (`frontmatter/zusammenfassung.tex`)
- [x] `references.bib` mit verifizierten Quellen befüllt (GR00T N1, SIMPLER, Flow Matching, π₀, π-RL, …)
- [x] Anhang an echte Datensatz-/SLURM-/Software-Fakten angepasst
- [x] Abgleich mit dem Projektstand vom 2026-08-25 (Lauf 3, `span`-Gate, Greif-Physik
      aufgelöst, geometrische Rekonstruktion, Widerruf des Vorzeichen-Fixes, Co-Training)
- [x] `latexmk` läuft lokal durch (LuaLaTeX + Biber sind installiert), keine undefinierten
      Referenzen, keine kritischen Overfull-Boxen
- [x] Verfasser eingetragen: Maximilian Berger, Matthias Späth, Luca Mücke
      (`\docAuthorA/B/C` in `preamble/settings.tex`; Titelblatt und ehrenwörtliche
      Erklärung geben alle drei einzeln aus)
- [ ] Restliche persönliche Daten in `preamble/settings.tex` (`\docMatrikelA/B/C`,
      `\docBetreuer`, `\docHochschule`, `\docFachbereich`, `\docStudiengang`,
      `\docAbgabedatum`)
- [ ] Optional: echte Abbildungen in `figures/` einbinden (statt der TikZ-Diagramme/W&B-Plot)
- [x] Ehrenwörtliche Erklärung geprüft (Formulierung „keine anderen als die angegebenen Quellen und Hilfsmittel“ ergänzt; gegen Vorlage der Hochschule abgleichen)

### Inhaltlich offen (hängt am Projekt, nicht am Dokument)

- Die simulationsseitigen Greifzahlen (0/20, Fingerspanne 19 % / 20,5 % / 27,6 %) sind mit
  korrigierter Geometrie **neu zu erheben** — siehe Abschnitt „Vorbehalt gegen alle
  simulationsseitigen Greifzahlen" in Kapitel 5.
- Sobald der Co-Training-Lauf gefahren ist, gehört sein Ergebnis in Kapitel 6/7.
