# LaTeX — Projektarbeit (Ausarbeitung)

> [!WARNING]
> **Der Inhalt ist bisher reiner Füll-/Platzhaltertext und noch NICHT projektspezifisch
> angepasst.** In diesem Ordner geht es aktuell nur darum, das **Layout grob aufzusetzen**
> (Dokumentklasse, Satzspiegel, Verzeichnisse, Kapitelgerüst, Literatur-/Abkürzungsapparat).
> Sämtliche Texte in `chapters/`, `frontmatter/` und `backmatter/` sind Blindtext bzw.
> generische Platzhalter und müssen vor der Abgabe vollständig durch den echten Projektinhalt
> ersetzt werden. Auch `references.bib`, Titelseite und Abkürzungen sind noch nicht final.

## Zweck

Gerüst für die schriftliche Ausarbeitung der Projektarbeit (Fine-Tuning von NVIDIA GR00T N1.6
auf dem Unitree G1 + DEX3-Hand). Stand: **nur Layout/Struktur** — der inhaltliche Teil folgt
später.

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

- [ ] Titelseite mit echten Projekt-/Personendaten füllen (`frontmatter/titelseite.tex`)
- [ ] Blindtext in allen `chapters/*.tex` durch echten Inhalt ersetzen
- [ ] Zusammenfassung schreiben (`frontmatter/zusammenfassung.tex`)
- [ ] `references.bib` mit den tatsächlich zitierten Quellen befüllen
- [ ] Abkürzungsverzeichnis und Abbildungen ergänzen
- [ ] Anhang und ehrenwörtliche Erklärung finalisieren
