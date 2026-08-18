# .latexmkrc — Latexmk-Konfiguration für LuaLaTeX + Biber + minted
#
# Aufruf: latexmk main.tex
# Voraussetzungen:
#   - TeX Live 2022+ mit LuaLaTeX, Biber und KOMA-Script
#   - Python 3 + Pygments:  pip install Pygments
#   - latexmk:              in TeX Live enthalten

# Python-Pfad unter Windows voranstellen — verhindert den Windows-Store-Stub-Konflikt,
# bei dem `python` auf einen Platzhalter zeigt und minted/latexminted scheitert.
# Der Ort ist rechnergebunden und steht deshalb NICHT mehr fest im Skript (bis 2026-08 war
# hier ein bestimmtes Benutzerprofil verdrahtet):
#   * Standard: die üblichen Installationsorte von Python 3.11-3.13 werden durchprobiert,
#     genommen wird der erste, der wirklich existiert;
#   * eigener Ort: Umgebungsvariable LATEX_PYTHON_DIR setzen.
# Unter Linux/macOS passiert hier bewusst nichts — dort liefert die Distribution python3 und
# pygments systemweit, latexmk findet sie über den normalen PATH.
if ($^O eq 'MSWin32') {
    my @candidates = $ENV{LATEX_PYTHON_DIR}
        ? ($ENV{LATEX_PYTHON_DIR})
        : map { "$ENV{LOCALAPPDATA}\\Programs\\Python\\$_" } qw(Python313 Python312 Python311);
    foreach my $dir (@candidates) {
        next unless -d $dir;
        $ENV{PATH} = "$dir;$dir\\Scripts;" . $ENV{PATH};
        last;
    }
}

# Engine: LuaLaTeX mit Shell-Escape (für minted)
$pdf_mode = 4;
$lualatex = 'lualatex -shell-escape -interaction=nonstopmode -synctex=1 %O %S';

# Biber als Backend für biblatex
$bibtex_use = 2;
$biber = 'biber %O %S';

# Hilfsdateien, die bei 'latexmk -c' gelöscht werden
$clean_ext = join(' ', qw(
  acn acr alg
  bbl bcf blg
  fdb_latexmk fls
  glg glo gls ist
  lof lot
  nav out
  run.xml
  snm synctex.gz
  toc xdv
  auxlock
));

# minted-Cache-Verzeichnis ebenfalls löschen
$pdflatex_silent_switch = '';
push @generated_exts, 'acn', 'acr', 'alg', 'glg', 'glo', 'gls', 'ist';

# SyncTeX für IDE-Integration (VS Code / TeXstudio)
$synctex = 1;

# Maximale Kompilierungsläufe (normalerweise reichen 3)
$max_repeat = 5;
