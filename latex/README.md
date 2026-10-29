# LaTeX — Projektarbeit (Ausarbeitung)

> [!NOTE]
> **Stand: 2026-10-29** (20 Seiten Hauptteil / 41 Seiten gesamt, baut ohne Fehler und ohne
> undefinierte Referenzen). Neu gegliedert nach [`docs/abgabe-plan.md`](../docs/abgabe-plan.md)
> Teil B: System, Schnittstellen und Anleitung zum Weiterarbeiten im Mittelpunkt, Ergebnisse
> als Kette bis Lauf 97 (N1.7, Orakel/Registrierung, Domain-Gap, Co-Training v3), Lessons
> Learned als eigenes Kapitel. **Noch offen:** Matrikelnummern A/B und Abgabedatum in
> `preamble/settings.tex`.

## Zweck

Schriftliche Ausarbeitung der Projektarbeit. Sie ist bewusst knapp und verweist für Befehle,
Parameter und Einzelwerte auf `docs/`; jede Zahl stammt aus
[`docs/ergebnisse/diagnose-chronik.md`](../docs/ergebnisse/diagnose-chronik.md) oder den CSVs
unter [`figures/daten/`](figures/daten/README.md).

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
│   ├── 01_einleitung.tex        #  Aufgabe, Ergebnis in fünf Sätzen, Aufbau, Beiträge
│   ├── 02_system.tex            #  Architekturbild, Plattform, GR00T N1.6/N1.7, Rechenumgebungen
│   ├── 03_schnittstellen.tex    #  Datensatz, Checkpoint, Policy-Server, Sim-Eval, Container, Roboter
│   ├── 04_anleitung.tex         #  Zugänge, Training, Checkpoint messen, häufige Fehler
│   ├── 05_ergebnisse.tex        #  Training → Closed Loop → Blocker 1/2 → Co-Training → N1.7
│   ├── 06_grenzen.tex           #  Stichprobe, Sim ≠ Roboter, Grenzen der Metriken
│   ├── 07_projektverlauf.tex    #  Verlauf, Fehler ohne Meldung, KI-Einsatz, Lessons Learned
│   └── 08_uebergabe.tex         #  offene Punkte, bekannte Defekte
├── backmatter/
│   ├── anhang.tex           # Datensatz, SLURM, Software, Trainingsdynamik, Szene
│   └── erklaerung.tex       # Ehrenwörtliche Erklärung ← prüfen
├── figures/                 # Abbildungen; plots/ aus make_figures.py (daten/*.csv)
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

## Abbildungen neu erzeugen

```bash
uv run --with matplotlib python latex/figures/make_figures.py   # aus dem Repo-Wurzelverzeichnis
```

Schreibt `figures/plots/*.pdf` aus `figures/daten/*.csv`. Die PDFs sind eingecheckt, damit
`make` ohne Python läuft.

## Offen

- [ ] `\docMatrikelA`, `\docMatrikelB`, `\docAbgabedatum` in `preamble/settings.tex`
- [ ] Ehrenwörtliche Erklärung gegen die Vorlage der Hochschule abgleichen
- [ ] Prüfen vor der Abgabe nach `docs/abgabe-plan.md` (Zahlen-Stichprobe, Fremdtest, Absatz-Test)
