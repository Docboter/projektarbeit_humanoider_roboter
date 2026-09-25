#!/usr/bin/env bash
# TL;DR: Prueft, dass jede Doku-Seite und jedes Skript ein TL;DR traegt; --list druckt alle als Uebersicht.
#
# check_tldr.sh — Huetet die TL;DR-Konvention (Host-Werkzeug, nie im Image).
#
# Konvention:
#   Doku (.md unter docs/, README.md, next-steps.md):
#       direkt unter der H1 ein Blockquote, erste Zeile beginnt mit  "> **TL;DR:** ..."
#   Skripte (.sh/.py/.ps1/.spec, Dockerfile*, docker-compose.yml unter Training/, Simulation/, tools/):
#       Kommentarzeile  "# TL;DR: ..."  in den ersten 3 Zeilen (nach Shebang bzw. "# syntax=");
#       eine Fortsetzungszeile "#   ..." ist erlaubt.
#
# Aufruf:
#   tools/check_tldr.sh                    # Pruefung; Exit 1, wenn etwas fehlt
#   tools/check_tldr.sh --list             # alle TL;DRs als Uebersicht
#   tools/check_tldr.sh --list docs/sim    # nur Pfade mit diesem Praefix
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mode=check
filter=""
for a in "$@"; do
  case "$a" in
    --list) mode=list ;;
    -h|--help) awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print; next } exit }' "$0"; exit 0 ;;
    *) filter="$a" ;;
  esac
done

doc_files() { find docs README.md next-steps.md -name '*.md' | sort; }
script_files() {
  # Auch die Skripte direkt im Wurzelverzeichnis (run.sh) — die Konvention gilt
  # dort genauso, und gerade der Einstiegspunkt sollte sein TL;DR nicht verpassen.
  { find Training Simulation tools -type f \
      \( -name '*.sh' -o -name '*.py' -o -name '*.ps1' -o -name '*.spec' \
         -o -name 'Dockerfile*' -o -name 'docker-compose.yml' \)
    find . -maxdepth 1 -type f -name '*.sh'
  } | sed 's|^\./||' | sort
}

# TL;DR-Text einer Datei (leer, wenn keins gefunden)
tldr_of() {
  local f="$1"
  case "$f" in
    *.md)
      head -12 "$f" | awk '
        /^> \*\*TL;DR:\*\*/ { on=1 }
        on && !/^>/        { exit }
        on { sub(/^> ?/, ""); sub(/^\*\*TL;DR:\*\* ?/, ""); printf "%s ", $0 }' ;;
    *)
      head -4 "$f" | awk '
        /^# TL;DR: /        { on=1; sub(/^# TL;DR: /, ""); printf "%s ", $0; next }
        on && /^#   /       { sub(/^#  */, ""); printf "%s ", $0; next }
        on                  { exit }' ;;
  esac
}

total=0; missing=0
while IFS= read -r f; do
  [[ -n "$filter" && "$f" != "$filter"* ]] && continue
  total=$((total + 1))
  t="$(tldr_of "$f")"; t="${t% }"
  if [[ -z "$t" ]]; then
    missing=$((missing + 1))
    if [[ $mode == list ]]; then printf '%s\n    (kein TL;DR)\n' "$f"; else echo "FEHLT  $f"; fi
  elif [[ $mode == list ]]; then
    printf '%s\n    %s\n' "$f" "$t"
  fi
done < <({ doc_files; script_files; })

if [[ $mode == check ]]; then
  echo "TL;DR vorhanden: $((total - missing))/$total Dateien"
  (( missing == 0 )) || exit 1
fi
