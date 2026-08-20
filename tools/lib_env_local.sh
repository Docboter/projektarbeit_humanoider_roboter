#!/usr/bin/env bash
# lib_env_local.sh — gemeinsames Laden der gitignorierten .env.local.
#
# Vorher stand dieser Block wortgleich in server_rl_run.sh und server_robocasa_ref_run.sh,
# und in den Trainings-Launchern fehlte er ganz — dort galt die in docs/portabilitaet.md
# beschriebene Vorrangregel also gar nicht. Diese Datei ist die eine Fassung davon.
#
# VORRANG (unveraendert):  explizite Umgebungsvariable  >  .env.local  >  Default im Skript
#   Damit das gilt, nutzt .env.local die Form  : "${VAR:=wert}"  — ein nacktes VAR=wert
#   wuerde eine bereits gesetzte Variable ueberschreiben.
#   `set -a` exportiert das Gesetzte, damit es Unterprozesse (docker, apptainer) erreicht.
#
# ZUSATZ fuer die Menuefuehrung: die Datei merkt sich, WELCHE Variablen aus .env.local
# kamen. Das Menue zeigt sie damit als "(aus .env.local)" an, statt sie erneut zu fragen —
# und genau das macht das Menue umso stiller, je besser .env.local gepflegt ist.
#
# Nutzung:
#   source "$REPO_DIR/tools/lib_env_local.sh"
#   env_local_load "$REPO_DIR"

[[ -n "${_ENV_LOCAL_LIB_LOADED:-}" ]] && return 0
_ENV_LOCAL_LIB_LOADED=1

# Namen der Variablen, die .env.local neu gesetzt hat (durch Leerzeichen getrennt).
ENV_LOCAL_PROVIDED=""
# Namen der Variablen, die der Aufrufer schon VOR dem Laden exportiert hatte —
# also das, was auf der Kommandozeile stand. Das Menue fragt sie nicht, sondern zeigt
# sie als "vorgegeben" an. Ohne diese Momentaufnahme koennte es sie nicht von den
# Skript-Defaults unterscheiden, die weiter unten im Konfigblock gesetzt werden.
ENV_PRESET=""
# Pfad der geladenen Datei — leer, wenn keine existiert.
ENV_LOCAL_FILE=""

env_local_load() {
  local repo_dir="${1:-${REPO_DIR:-.}}"
  local file="$repo_dir/.env.local"

  # Momentaufnahme der bereits exportierten Namen. Passiert IMMER — auch ohne .env.local,
  # denn das Menue braucht sie in jedem Fall.
  local before after
  before="$(compgen -e | sort)"
  ENV_PRESET=" $(printf '%s' "$before" | tr '\n' ' ') "

  [[ -f "$file" ]] || return 0

  set -a
  # shellcheck source=/dev/null
  source "$file"
  set +a

  after="$(compgen -e | sort)"
  ENV_LOCAL_PROVIDED="$(comm -13 <(printf '%s\n' "$before") <(printf '%s\n' "$after") | tr '\n' ' ')"
  ENV_LOCAL_FILE="$file"
  return 0
}

# Kam <VAR> aus .env.local?  Rueckgabewert 0 = ja.
env_local_provided() {
  [[ " $ENV_LOCAL_PROVIDED " == *" $1 "* ]]
}

# War <VAR> schon vor dem Laden exportiert (also vom Aufrufer gesetzt)?
env_preset() {
  [[ "$ENV_PRESET" == *" $1 "* ]]
}
