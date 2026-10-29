#!/usr/bin/env python3
# TL;DR: Prüft alle relativen Markdown-Links und -Anker im Repo (ohne app/, graphify-out/); Exit 1 bei kaputten Links.
"""Relative Links ([text](pfad#anker)) in *.md prüfen: Datei vorhanden, Anker vorhanden.

Anker werden wie auf GitHub gebildet: Kleinbuchstaben, Satzzeichen außer '-' und '_' entfernt,
Leerzeichen zu '-', doppelte Überschriften bekommen '-1', '-2', … angehängt. Externe Links
(http, mailto) werden nicht geprüft.

Aufruf: python3 tools/check_links.py [--quiet]
"""

from pathlib import Path
import re
import sys
import unicodedata
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {"app", "graphify-out", "node_modules", ".git", "_minted", "data"}
LINK_RE = re.compile(r"(?<!!)\[(?:[^\]\[]|\[[^\]]*\])*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
HTML_ANCHOR_RE = re.compile(r"<a\s+(?:name|id)=\"([^\"]+)\"")


def slugify(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)  # HTML-Tags
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # Links -> Linktext
    text = text.replace("`", "").lower()
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in "-_ " or cat[0] in ("L", "N") or (cat == "Mn" and ch != "\ufe0f"):
            out.append(ch)
    return "".join(out).replace(" ", "-")


def strip_code(lines, inline=True):
    """Zeilen in ```-Blöcken leeren; mit inline=True auch Inline-Code entfernen."""
    in_fence = False
    for line in lines:
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            yield ""
            continue
        if in_fence:
            yield ""
        else:
            yield re.sub(r"`[^`]*`", "", line) if inline else line


_anchor_cache: dict[Path, set[str]] = {}


def anchors(path: Path) -> set[str]:
    if path not in _anchor_cache:
        seen: dict[str, int] = {}
        result = set()
        for line in strip_code(path.read_text(encoding="utf-8").splitlines(), inline=False):
            result.update(HTML_ANCHOR_RE.findall(line))
            m = HEADING_RE.match(line)
            if not m:
                continue
            slug = slugify(m.group(2))
            n = seen.get(slug, 0)
            seen[slug] = n + 1
            result.add(slug if n == 0 else f"{slug}-{n}")
        _anchor_cache[path] = result
    return _anchor_cache[path]


def md_files():
    for p in ROOT.rglob("*.md"):
        if not any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
            yield p


def main() -> int:
    quiet = "--quiet" in sys.argv
    errors = 0
    files = sorted(md_files())
    for md in files:
        for lineno, line in enumerate(strip_code(md.read_text(encoding="utf-8").splitlines()), 1):
            for target in LINK_RE.findall(line) + IMG_RE.findall(line):
                if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I):
                    continue  # http:, mailto:, …
                path_part, _, anchor = target.partition("#")
                path_part = path_part.split("?")[0]
                dest = (md.parent / unquote(path_part)).resolve() if path_part else md
                rel = md.relative_to(ROOT)
                if not dest.exists():
                    print(f"{rel}:{lineno}: Ziel fehlt: {target}")
                    errors += 1
                    continue
                if anchor and dest.suffix == ".md" and unquote(anchor) not in anchors(dest):
                    print(f"{rel}:{lineno}: Anker fehlt: {target}")
                    errors += 1
    if not quiet or errors:
        print(f"{len(files)} Markdown-Dateien geprüft, {errors} kaputte Links.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
