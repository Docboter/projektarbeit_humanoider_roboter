# .latexmkrc — Latexmk-Konfiguration für LuaLaTeX + Biber + minted
#
# Aufruf: latexmk main.tex
# Voraussetzungen:
#   - TeX Live 2022+ mit LuaLaTeX, Biber und KOMA-Script
#   - Python 3 + Pygments:  pip install Pygments
#   - latexmk:              in TeX Live enthalten

# Python-Pfad explizit setzen (verhindert Windows-Store-Stub-Konflikt)
$ENV{PATH} = 'C:\Users\mueck\AppData\Local\Programs\Python\Python312;'
           . 'C:\Users\mueck\AppData\Local\Programs\Python\Python312\Scripts;'
           . $ENV{PATH};

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
