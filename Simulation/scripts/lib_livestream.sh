#!/usr/bin/env bash
# TL;DR: Gesourcte Bibliothek: gemeinsame WebRTC-Livestream-Logik für Entrypoints + server_rl_run.sh.
# lib_livestream.sh — gemeinsame Logik für die LIVE-Variante („Spur A" des
# Livestream-Plans): Isaac Sim streamt seinen 3D-Viewport per WebRTC, und auf dem
# Arbeitsrechner öffnet der native **Isaac Sim WebRTC Streaming Client** die
# Simulationsumgebung, als stünde sie lokal.
#
# Wird gesourct von:
#   - /scripts/entrypoint_sim.sh, _baseline.sh, _replay.sh   (im Container)
#   - Simulation/server_rl_run.sh                            (auf dem HOST)
# Deshalb darf hier nichts stehen, was zwingend einen Container voraussetzt: die
# Versionserkennung fällt ohne Isaac-Sim-Installation still auf „beide Settings-Pfade"
# zurück (s. livestream_kit_args).
#
# Warum eine gemeinsame Datei statt drei Kopien: entrypoint_sim.sh und
# entrypoint_baseline.sh trugen bis 2026-08-13 fast wortgleiche Livestream-Blöcke —
# genau das Muster, das bei der nächsten Korrektur auseinanderläuft (Risiko-Tabelle
# in docs/weiterfuehrend/livestream-plan.md §8).
#
# Env-Vars (alle optional, Defaults unten):
#   LIVESTREAM              0=aus (default), 1=WebRTC öffentlich, 2=WebRTC privat/lokal
#   LIVESTREAM_PORT         Signaling-Port, TCP (default 49100)
#   LIVESTREAM_MEDIA_PORT   Medien-Port, UDP  (default 47998)
#   LIVE_KEEP_VIDEO         1 = trotz Live-Ansicht zusätzlich MP4s schreiben (default 0)
#   PUBLIC_IP               Öffentliche IP; nur im Modus 1 relevant, sonst ungenutzt
#   LIVESTREAM_HOST_ADDR    Adresse, die im Verbindungshinweis erscheint (der Container
#                           kennt die Host-IP nicht — server_rl_run.sh reicht sie durch)
#   LIVESTREAM_SETTINGS_STYLE  auto|new|old|both  (default auto) — s. livestream_kit_args
#   LIVESTREAM_KIT_ARGS     manueller Override der Kit-Settings (gewinnt über alles)

# ── Logging-Helfer nur definieren, wenn der Aufrufer keine hat ────────────────
# entrypoint_*.sh und server_rl_run.sh bringen log/ok/warn/err selbst mit; die hier
# sind nur für den Fall, dass die Lib mal isoliert gesourct wird.
declare -F log  >/dev/null || log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
declare -F ok   >/dev/null || ok()   { printf '\033[1;32m v \033[0m %s\n' "$*"; }
declare -F warn >/dev/null || warn() { printf '\033[1;33m ! \033[0m %s\n' "$*"; }
declare -F err  >/dev/null || err()  { printf '\033[1;31m!! \033[0m %s\n' "$*" >&2; }

# Ausgabe von livestream_app_flags. Vorbelegt, damit `set -u` beim Aufrufer nicht
# zuschlägt, falls jemand das Array vor dem ersten Aufruf liest.
LS_APP_FLAGS=()

# ── Initialisierung ───────────────────────────────────────────────────────────
# Normalisiert alle Variablen und ermittelt PUBLIC_IP — aber NUR im Modus 1.
# (Defekt D4: im privaten Netz ist PUBLIC_IP bedeutungslos, und der curl-Aufruf
# kostete dort im Institutsnetz bis zu 10 s Startverzögerung.)
livestream_init() {
    LIVESTREAM="${LIVESTREAM:-0}"
    LIVESTREAM_PORT="${LIVESTREAM_PORT:-49100}"
    LIVESTREAM_MEDIA_PORT="${LIVESTREAM_MEDIA_PORT:-47998}"
    LIVE_KEEP_VIDEO="${LIVE_KEEP_VIDEO:-0}"

    if [[ "$LIVESTREAM" == "1" && -z "${PUBLIC_IP:-}" ]]; then
        PUBLIC_IP="$(curl -s --max-time 10 ifconfig.me || true)"
    fi

    export LIVESTREAM LIVESTREAM_PORT LIVESTREAM_MEDIA_PORT LIVE_KEEP_VIDEO
    [[ -n "${PUBLIC_IP:-}" ]] && export PUBLIC_IP
    return 0
}

livestream_active() { [[ "${LIVESTREAM:-0}" != "0" ]]; }

# ── Isaac-Sim-Hauptversion ────────────────────────────────────────────────────
# Bestimmt, welche Kit-Settings-Pfade gültig sind. Leer, wenn keine Isaac-Sim-
# Installation sichtbar ist (z. B. beim Sourcen auf dem Host).
_livestream_isaac_major() {
    local vf
    for vf in "${ISAACLAB_PATH:-/workspace/isaaclab}/_isaac_sim/VERSION" \
              "${ISAAC_SIM_PATH:-/isaac-sim}/VERSION"; do
        if [[ -r "$vf" ]]; then
            head -n1 "$vf" | grep -oE '^[0-9]+' && return 0
        fi
    done
    echo ""
}

# ── Kit-Settings (Defekt D2) ──────────────────────────────────────────────────
# Isaac Sim ≤ 5.x:  --/app/livestream/{port,publicEndpointAddress}
# Isaac Sim 6.0:    --/exts/omni.kit.livestream.app/primaryStream/{signalPort,streamPort,publicIp}
#
# Kit ignoriert unbekannte Settings STILL — ein falscher Pfad führt also nicht zu einem
# Fehler, sondern zu einem Stream auf dem Default-Port. Genau deshalb:
#   - `auto` (Default) wählt anhand der VERSION-Datei,
#   - ohne erkennbare Version werden BEIDE Pfade gesetzt (der jeweils falsche verpufft
#     folgenlos, der richtige greift),
#   - `LIVESTREAM_KIT_ARGS` überschreibt alles, falls die Praxis etwas Drittes zeigt.
#
# Ausgegeben wird nur, was tatsächlich nötig ist: bei Default-Ports im privaten Netz
# (der Server-Fall) bleibt der String LEER. Das entschärft zugleich Defekt D3 — der
# AppLauncher injiziert im Modus 1 selbst `--/app/livestream/port=49100`, und ein
# zweiter, abweichender Port im --kit_args ließe die Reihenfolge entscheiden.
livestream_kit_args() {
    if [[ -n "${LIVESTREAM_KIT_ARGS:-}" ]]; then
        printf '%s' "$LIVESTREAM_KIT_ARGS"
        return 0
    fi

    local port="${LIVESTREAM_PORT:-49100}"
    local media="${LIVESTREAM_MEDIA_PORT:-47998}"
    local pub=""
    [[ "${LIVESTREAM:-0}" == "1" && -n "${PUBLIC_IP:-}" ]] && pub="$PUBLIC_IP"

    # Nichts zu setzen → nichts ausgeben (s. D3 oben).
    if [[ "$port" == "49100" && "$media" == "47998" && -z "$pub" ]]; then
        printf ''
        return 0
    fi

    local style="${LIVESTREAM_SETTINGS_STYLE:-auto}"
    if [[ "$style" == "auto" ]]; then
        local major; major="$(_livestream_isaac_major)"
        if   [[ -z "$major" ]];        then style="both"
        elif [[ "$major" -ge 6 ]];     then style="new"
        else                                style="old"
        fi
    fi

    local args=()
    case "$style" in
        new|both)
            args+=( "--/exts/omni.kit.livestream.app/primaryStream/signalPort=$port" )
            args+=( "--/exts/omni.kit.livestream.app/primaryStream/streamPort=$media" )
            [[ -n "$pub" ]] && args+=( "--/exts/omni.kit.livestream.app/primaryStream/publicIp=$pub" )
            ;;
    esac
    case "$style" in
        old|both)
            args+=( "--/app/livestream/port=$port" )
            [[ -n "$pub" ]] && args+=( "--/app/livestream/publicEndpointAddress=$pub" )
            ;;
    esac
    printf '%s' "${args[*]}"
}

# ── App-Flags ─────────────────────────────────────────────────────────────────
# Setzt LS_APP_FLAGS auf entweder ( --headless ) oder ( --livestream N [--kit_args …] ).
#
# Bei aktivem Livestream darf --headless NICHT zusätzlich gesetzt werden: Isaac Lab wählt
# dann das falsche Experience-File (IsaacLab#381) und der Stream bleibt schwarz. Jeder
# Livestream-Modus impliziert ohnehin Headless.
livestream_app_flags() {
    LS_APP_FLAGS=()
    if livestream_active; then
        LS_APP_FLAGS+=( --livestream "${LIVESTREAM}" )
        local kit; kit="$(livestream_kit_args)"
        [[ -n "$kit" ]] && LS_APP_FLAGS+=( --kit_args "$kit" )
    else
        LS_APP_FLAGS+=( --headless )
    fi
    return 0
}

# ── Video-Verzeichnis (die „statt Videos"-Regel) ──────────────────────────────
# Gibt den übergebenen Pfad zurück — oder einen LEEREN String, wenn die Live-Variante
# läuft. Die Eval-/Replay-Skripte schalten bei leerem --video-dir das Frame-Sammeln,
# die MP4-Ausgabe und den rgb_array-Render-Mode ab (`record = bool(args.video_dir)`).
# LIVE_KEEP_VIDEO=1 erzwingt beides gleichzeitig.
livestream_video_dir() {
    local requested="$1"
    if livestream_active && [[ "${LIVE_KEEP_VIDEO:-0}" != "1" ]]; then
        printf ''
    else
        printf '%s' "$requested"
    fi
}

# ── Verbindungshinweis ────────────────────────────────────────────────────────
# Ein Argument: die Adresse, unter der der Server vom Arbeitsrechner aus erreichbar ist.
# Leer → Platzhalter. Der Container kennt die Host-IP nicht (er sähe nur seine 172.x-
# Bridge-Adresse), deshalb reicht server_rl_run.sh sie als LIVESTREAM_HOST_ADDR durch.
livestream_banner() {
    livestream_active || return 0
    local addr="${1:-${LIVESTREAM_HOST_ADDR:-}}"
    [[ -z "$addr" && "${LIVESTREAM:-0}" == "1" ]] && addr="${PUBLIC_IP:-}"
    [[ -z "$addr" ]] && addr="<server-ip>"

    local modus="privat/lokal"; [[ "$LIVESTREAM" == "1" ]] && modus="öffentlich"
    log "LIVE-Variante AKTIV — WebRTC-Viewport (Modus $LIVESTREAM, $modus)"
    echo "    Auf dem Arbeitsrechner öffnen:"
    echo "      1. App 'Isaac Sim WebRTC Streaming Client' starten"
    echo "         (Download: https://docs.isaacsim.omniverse.nvidia.com → Livestream Clients)"
    echo "      2. Server: ${addr}:${LIVESTREAM_PORT}   → Connect"
    echo "    Offen sein müssen:  ${LIVESTREAM_PORT}/tcp (Signaling) + ${LIVESTREAM_MEDIA_PORT}/udp (Video)"
    echo "    Der Viewport erscheint, sobald Isaac Sim die Szene geladen hat (dauert ~1 min)."
    if [[ "${LIVE_KEEP_VIDEO:-0}" == "1" ]]; then
        warn "LIVE_KEEP_VIDEO=1 — MP4s werden zusätzlich geschrieben (kostet Render-Kopien)."
    else
        log "MP4-Aufzeichnung ist AUS (Live statt Video). Beides zugleich: LIVE_KEEP_VIDEO=1"
    fi
    # Der Stream ist ungeschützt — im VPN/Institutsnetz vertretbar, auf einer öffentlichen
    # Cloud-IP nicht. Deshalb genau dann warnen, wenn Modus 1 gewählt wurde.
    if [[ "$LIVESTREAM" == "1" ]]; then
        warn "Modus 1 exponiert den Viewport ohne Authentifizierung ins öffentliche Netz."
        warn "  Im Instituts-/VPN-Netz stattdessen LIVESTREAM=2 nutzen."
        warn "  vast.ai: LIVESTREAM_PORT MUSS dem extern gemappten Port entsprechen"
        warn "           (intern==extern), sonst zeigt die SDP-Aushandlung ins Leere."
    fi
    return 0
}
