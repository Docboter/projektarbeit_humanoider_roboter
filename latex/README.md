# LaTeX — Projektarbeit (Ausarbeitung)

> [!NOTE]
> **Stand: 2026-08-25** (69 Seiten, baut sauber durch). Der inhaltliche Teil ist ausgearbeitet
> und mit der Projektdokumentation unter [`docs/`](../docs/README.md) abgeglichen — einschließlich
> der Diagnoseläufe 35–52. Die Kapitel beschreiben das *Was/Wann/Wie/Warum* des Projekts und
> verweisen für Reproduktionsdetails (Kommandos, Variablen) auf die jeweilige `docs/`-Datei, um
> Doppelungen zu vermeiden. **Noch offen:** die persönlichen Titelseiten-Daten (Verfasser,
> Matrikelnummer, Betreuer, Hochschule, Studiengang, Abgabedatum) in `preamble/settings.tex`
> sowie ggf. das Einbinden echter Abbildungen statt der TikZ-Diagramme in `figures/`.

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
│   ├── 01_einleitung.tex                #  Motivation, Ziele, Abgrenzung
│   ├── 02_grundlagen.tex                #  VLA, Flow Matching, Domain-Gap
│   ├── 03_methoden.tex                  #  Infrastruktur, Datensatz, Training, Eval
│   ├── 04_ergebnisse.tex                #  Läufe 1–3, Diagnosekette, Geometrie
│   ├── 05_diskussion.tex                #  Einordnung, Messartefakte, Limitierungen
│   ├── 06_weiterfuehrende_arbeiten.tex  #  Lokomotion, Co-Training, RL
│   └── 07_fazit.tex                     #  Fazit und Ausblick
├── backmatter/
│   ├── anhang.tex           # Datensatz, SLURM, Software, Trainingsdynamik, Szene
│   └── erklaerung.tex       # Ehrenwörtliche Erklärung ← prüfen
├── figures/                 # Abbildungen
├── references.bib           # Literatur (18 Einträge, 16 zitiert)
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

- [x] Inhalt aller `chapters/*.tex` ausgearbeitet (01 Einleitung … 07 Fazit)
- [x] Zusammenfassung / Abstract geschrieben (`frontmatter/zusammenfassung.tex`)
- [x] `references.bib` mit verifizierten Quellen befüllt (GR00T N1, SIMPLER, Flow Matching, π₀, π-RL, …)
- [x] Anhang an echte Datensatz-/SLURM-/Software-Fakten angepasst
- [x] Abgleich mit dem Projektstand vom 2026-08-25 (Lauf 3, `span`-Gate, Greif-Physik
      aufgelöst, geometrische Rekonstruktion, Widerruf des Vorzeichen-Fixes, Co-Training)
- [x] `latexmk` läuft lokal durch (LuaLaTeX + Biber sind installiert), keine undefinierten
      Referenzen, keine kritischen Overfull-Boxen
- [ ] Persönliche Daten in `preamble/settings.tex` (`\docAuthor`, `\docMatrikel`, `\docBetreuer`,
      `\docHochschule`, `\docFachbereich`, `\docStudiengang`, `\docAbgabedatum`)
- [ ] Optional: echte Abbildungen in `figures/` einbinden (statt der TikZ-Diagramme/W&B-Plot)
- [ ] Ehrenwörtliche Erklärung prüfen (`backmatter/erklaerung.tex`)

### Inhaltlich offen (hängt am Projekt, nicht am Dokument)

- Die simulationsseitigen Greifzahlen (0/20, Fingerspanne 19 % / 20,5 % / 27,6 %) sind mit
  korrigierter Geometrie **neu zu erheben** — siehe Abschnitt „Vorbehalt gegen alle
  simulationsseitigen Greifzahlen" in Kapitel 5.
- Sobald der Co-Training-Lauf gefahren ist, gehört sein Ergebnis in Kapitel 6/7.
