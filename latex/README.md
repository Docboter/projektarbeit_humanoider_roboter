# LaTeX — Projektarbeit (Ausarbeitung)

> [!NOTE]
> **Der inhaltliche Teil ist ausgearbeitet** und basiert auf der Projektdokumentation unter
> [`docs/`](../docs/README.md). Die Kapitel beschreiben das *Was/Wann/Wie/Warum* des Projekts
> und verweisen für Reproduktionsdetails (Kommandos, Variablen) auf die jeweilige `docs/`-Datei,
> um Doppelungen zu vermeiden. **Noch offen:** die persönlichen Titelseiten-Daten (Verfasser,
> Matrikelnummer, Betreuer, Hochschule, Studiengang, Abgabedatum) in `preamble/settings.tex`
> sowie ggf. das Einbinden echter Abbildungen (W&B-Plots, Architektur-Diagramm) in `figures/`.

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
│   ├── titelseite.tex       # Titelseite        ← Platzhalter
│   └── zusammenfassung.tex  # Zusammenfassung    ← Platzhalter
├── chapters/
│   ├── 01_einleitung.tex    #  ← Platzhalter / Blindtext
│   ├── 02_grundlagen.tex    #  ← Platzhalter / Blindtext
│   ├── 03_methoden.tex      #  ← Platzhalter / Blindtext
│   ├── 04_ergebnisse.tex    #  ← Platzhalter / Blindtext
│   ├── 05_diskussion.tex    #  ← Platzhalter / Blindtext
│   └── 06_fazit.tex         #  ← Platzhalter / Blindtext
├── backmatter/
│   ├── anhang.tex           # Anhang             ← Platzhalter
│   └── erklaerung.tex       # Ehrenwörtliche Erklärung ← Platzhalter
├── figures/                 # Abbildungen
├── references.bib           # Literatur          ← Platzhalter
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
- [ ] Persönliche Daten in `preamble/settings.tex` (`\docAuthor`, `\docMatrikel`, `\docBetreuer`,
      `\docHochschule`, `\docFachbereich`, `\docStudiengang`, `\docAbgabedatum`)
- [ ] Optional: echte Abbildungen in `figures/` einbinden (statt der TikZ-Diagramme/W&B-Plot)
- [ ] Ehrenwörtliche Erklärung prüfen (`backmatter/erklaerung.tex`)
- [ ] Einmal `latexmk` durchlaufen lassen (lokal kein TeX installiert — siehe „Kompilieren")
