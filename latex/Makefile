# Makefile für LaTeX-Projektarbeit
# Voraussetzungen: latexmk, lualatex, biber, Python + Pygments

MAIN   := main
TEXSRC := $(shell find . -name "*.tex" -not -path "./_minted*")
BIBSRC := references.bib

.PHONY: all watch clean distclean help

all: $(MAIN).pdf

$(MAIN).pdf: $(TEXSRC) $(BIBSRC)
	latexmk $(MAIN).tex

# Kontinuierliche Kompilierung bei Dateiänderungen (Ctrl+C zum Beenden)
watch:
	latexmk -pvc $(MAIN).tex

# Hilfsdateien löschen (PDF bleibt erhalten)
clean:
	latexmk -c $(MAIN).tex
	rm -rf _minted-$(MAIN)/

# Alle generierten Dateien löschen (einschließlich PDF)
distclean:
	latexmk -C $(MAIN).tex
	rm -rf _minted-$(MAIN)/

help:
	@echo "Verfügbare Befehle:"
	@echo "  make           — PDF kompilieren (LuaLaTeX + Biber)"
	@echo "  make watch     — Kontinuierliche Kompilierung bei Änderungen"
	@echo "  make clean     — Hilfsdateien löschen (PDF bleibt)"
	@echo "  make distclean — Alle generierten Dateien löschen"
	@echo ""
	@echo "Voraussetzungen:"
	@echo "  - TeX Live 2022+ (lualatex, biber, latexmk)"
	@echo "  - Python 3 + Pygments: pip install Pygments"
