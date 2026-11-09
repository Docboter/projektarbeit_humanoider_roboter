#!/usr/bin/env bash
# TL;DR: Baut die Projektkarte in allen Varianten (A4-PDF, Poster A1, Folien 16:9, SVG + PNG fürs Repo) aus den TikZ-Quellen dieses Ordners; Zwischendateien landen in build/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build

tex() { lualatex -interaction=nonstopmode -halt-on-error -output-directory=build "$@" >/dev/null \
          || { echo "Fehler, siehe build/*.log" >&2; exit 1; }; }

echo "Karte (voll, kurz) ..."
tex -jobname=karte_voll '\def\kartevariante{voll}\input{karte.tex}'
tex -jobname=karte_kurz '\def\kartevariante{kurz}\input{karte.tex}'

for doc in projektkarte_a4 projektkarte_poster_a1 projektkarte_folien; do
  echo "$doc ..."
  tex "$doc.tex"; tex "$doc.tex"     # zweiter Lauf: remember picture
  cp "build/$doc.pdf" .
done

echo "SVG + PNG (README) ..."
dvisvgm --pdf --no-fonts --exact-bbox --output=projektkarte.svg projektkarte_a4.pdf >/dev/null 2>&1
pdftoppm -r 200 -png -singlefile projektkarte_a4.pdf projektkarte
echo "fertig: $(ls -1 projektkarte*.{pdf,svg,png} | tr '\n' ' ')"
