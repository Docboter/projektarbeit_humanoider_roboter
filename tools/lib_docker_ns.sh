#!/usr/bin/env bash
# TL;DR: Gesourcte Host-Bibliothek — Docker-Hub-Konto (DOCKER_NAMESPACE) fuer Build/Push erfragen statt festschreiben.
# lib_docker_ns.sh — welches Docker-Hub-Konto baut und pusht die Images?
#
# Bis 2026-09 stand ueberall fest "lucam03/…". Ziehen (pull) kann das jeder, denn die Images
# sind oeffentlich. Pushen kann aber nur, wer Schreibrechte auf lucam03 hat — alle anderen
# liefen erst nach einem 30-Minuten-Build in ein "denied: requested access to the resource".
#
# Regel seitdem:
#   Lesen  (Launcher: pull/run)   DOCKER_NAMESPACE, sonst das Team-Konto lucam03.
#   Schreiben (update_*image.sh)   DOCKER_NAMESPACE MUSS vom Nutzer kommen — Umgebung,
#                                  .env.local oder Rueckfrage am Terminal. Ohne Terminal
#                                  bricht der Build VOR dem Bauen ab, statt still nach
#                                  lucam03 zu pushen.
# Wer DOCKER_NAMESPACE in .env.local traegt, zieht damit auch in den Launchern seine eigenen
# Images — Build und Start passen so automatisch zusammen.
#
# Nutzung (nach lib_env_local.sh):
#   source "$REPO_DIR/tools/lib_docker_ns.sh"
#   docker_ns_require_push "$REPO_DIR" || exit 1     # setzt + exportiert DOCKER_NAMESPACE
#   docker_ns_check_login                            # warnt / startet docker login

[[ -n "${_DOCKER_NS_LIB_LOADED:-}" ]] && return 0
_DOCKER_NS_LIB_LOADED=1

# Team-Konto: nur noch Default fuers LESEN, nie mehr stillschweigend fuers Schreiben.
DOCKER_NAMESPACE_TEAM="lucam03"

_dns_say()  { printf '%s\n' "$*" >&2; }
_dns_warn() { printf '\033[1;33m  ! \033[0m%s\n' "$*" >&2; }
_dns_err()  { printf '\033[1;31m!! \033[0m%s\n' "$*" >&2; }

# Bei Docker Hub angemeldeter Benutzer — best effort, gibt nie ein Passwort aus.
# Neuere Docker-Versionen zeigen ihn nicht mehr in `docker info`; dann wird der
# Credential-Helper bzw. der auths-Eintrag aus ~/.docker/config.json gelesen.
docker_hub_login_user() {
  local u
  u="$(docker info 2>/dev/null | sed -n 's/^ *Username: *//p' | head -1)"
  if [[ -n "$u" ]]; then printf '%s\n' "$u"; return 0; fi
  local cfg="${DOCKER_CONFIG:-$HOME/.docker}/config.json"
  [[ -f "$cfg" ]] && command -v python3 >/dev/null 2>&1 || return 0
  python3 - "$cfg" <<'PY' 2>/dev/null || true
import base64, json, subprocess, sys
cfg = json.load(open(sys.argv[1]))
keys = ["https://index.docker.io/v1/", "index.docker.io", "docker.io", "registry-1.docker.io"]
helpers = cfg.get("credHelpers", {})
for k in keys:
    helper = helpers.get(k) or cfg.get("credsStore")
    if helper:
        try:
            out = subprocess.run(["docker-credential-" + helper, "get"], input=k,
                                 capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                user = json.loads(out.stdout).get("Username", "")
                if user:
                    print(user); sys.exit(0)
        except Exception:
            pass
    auth = cfg.get("auths", {}).get(k, {}).get("auth")
    if auth:
        print(base64.b64decode(auth).decode().split(":", 1)[0]); sys.exit(0)
PY
}

# Gueltiger Image-Praefix? Docker-Hub-Konto (kleingeschrieben) oder Registry-Pfad wie
# ghcr.io/<konto>.
docker_ns_valid() {
  [[ "$1" =~ ^[a-z0-9][a-z0-9._-]*(/[a-z0-9][a-z0-9._-]*)*$ || "$1" =~ ^[a-z0-9.-]+(:[0-9]+)?/[a-z0-9][a-z0-9._/-]*$ ]]
}

# Stellt sicher, dass DOCKER_NAMESPACE fuer einen Push gesetzt ist.
#   $1 = Repo-Verzeichnis (fuer das Angebot, den Wert in .env.local zu merken)
# Rueckgabe 1, wenn kein Konto feststeht (kein Terminal, leere Eingabe).
docker_ns_require_push() {
  local repo_dir="${1:-.}"
  if [[ -n "${DOCKER_NAMESPACE:-}" ]]; then
    docker_ns_valid "$DOCKER_NAMESPACE" || {
      _dns_err "DOCKER_NAMESPACE='$DOCKER_NAMESPACE' ist kein gueltiger Image-Praefix (klein, z. B. 'mmuster')."
      return 1; }
    export DOCKER_NAMESPACE
    return 0
  fi
  if [[ ! -t 0 || ! -t 2 ]]; then
    _dns_err "Kein Docker-Hub-Konto fuer den Push angegeben — es wird nicht mehr still nach"
    _dns_err "'$DOCKER_NAMESPACE_TEAM' gepusht (dafuer braucht es dort Schreibrechte)."
    _dns_err "  Einmalig:   DOCKER_NAMESPACE=<dein-konto> $0 …"
    _dns_err "  Dauerhaft:  : \"\${DOCKER_NAMESPACE:=<dein-konto>}\"  in .env.local"
    _dns_err "  Nur lokal:  --skip-push"
    return 1
  fi
  local sug ans
  sug="$(docker_hub_login_user || true)"
  _dns_say ""
  _dns_say "  Auf welches Docker-Hub-Konto soll gepusht werden?  (DOCKER_NAMESPACE)"
  _dns_say "  Du brauchst dort Schreibrechte. Das Image heisst danach <konto>/projekt-humanoider-roboter…"
  [[ -n "$sug" ]] && _dns_say "  Angemeldet ist gerade: $sug"
  while :; do
    read -r -p "  Konto${sug:+ [$sug]}: " ans </dev/tty || return 1
    ans="${ans:-$sug}"
    ans="$(printf '%s' "$ans" | tr -d '[:space:]')"
    if [[ -z "$ans" ]]; then _dns_warn "Ohne Konto kein Push (Abbruch: Ctrl-C, nur bauen: --skip-push)."; continue; fi
    if docker_ns_valid "$ans"; then break; fi
    _dns_warn "'$ans' ist kein gueltiger Image-Praefix (Kleinbuchstaben, Ziffern, . _ -)."
  done
  DOCKER_NAMESPACE="$ans"
  export DOCKER_NAMESPACE
  local envf="$repo_dir/.env.local" yn
  if ! grep -qs 'DOCKER_NAMESPACE' "$envf"; then
    read -r -p "  In .env.local merken (dann fragt keiner mehr, und die Launcher ziehen es auch)? [J/n] " yn </dev/tty || yn=n
    if [[ ! "$yn" =~ ^[nN] ]]; then
      local fresh=0; [[ -f "$envf" ]] || fresh=1   # vor dem >>, das die Datei anlegt
      {
        (( fresh )) && printf '# .env.local — rechnerspezifisch, gitignoriert (Vorlage: .env.local.example)\n'
        printf '\n# Docker-Hub-Konto fuer eigene Images (Build/Push und Pull der Launcher)\n'
        printf ': "${DOCKER_NAMESPACE:=%s}"\n' "$ans"
      } >> "$envf"
      _dns_say "  -> in $envf eingetragen."
    fi
  fi
  return 0
}

# Warnt, wenn der Docker-Hub-Login fehlt oder nicht zum Konto passt. Ohne erkennbaren Login
# und am Terminal wird `docker login` gestartet. Nur fuer Docker Hub (kein Registry-Pfad).
docker_ns_check_login() {
  [[ "${DOCKER_NAMESPACE:-}" == */* ]] && return 0
  local user; user="$(docker_hub_login_user || true)"
  if [[ -z "$user" ]]; then
    if [[ -t 0 ]]; then
      _dns_warn "Kein Docker-Hub-Login erkannt — starte 'docker login' (Konto mit Schreibrechten auf '$DOCKER_NAMESPACE')."
      docker login </dev/tty || { _dns_err "docker login fehlgeschlagen."; return 1; }
    else
      _dns_warn "Kein Docker-Hub-Login erkannt — der Push nach '$DOCKER_NAMESPACE' scheitert vermutlich (vorher: docker login)."
    fi
  elif [[ "$user" != "$DOCKER_NAMESPACE" ]]; then
    _dns_warn "Angemeldet als '$user', Push geht nach '$DOCKER_NAMESPACE' — klappt nur mit Schreibrechten dort (Organisation/Team)."
  fi
  return 0
}
