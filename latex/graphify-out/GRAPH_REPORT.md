# Graph Report - latex  (2026-08-26)

## Corpus Check
- 1 files · ~161,142 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 7 nodes · 6 edges · 2 communities (1 shown, 1 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e966b09d`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- LaTeX — Projektarbeit (Ausarbeitung)
- Nächste Schritte (TODO)

## God Nodes (most connected - your core abstractions)
1. `LaTeX — Projektarbeit (Ausarbeitung)` - 5 edges
2. `Nächste Schritte (TODO)` - 2 edges
3. `Zweck` - 1 edges
4. `Struktur` - 1 edges
5. `Kompilieren` - 1 edges
6. `Inhaltlich offen (hängt am Projekt, nicht am Dokument)` - 1 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Communities (2 total, 1 thin omitted)

### Community 0 - "LaTeX — Projektarbeit (Ausarbeitung)"
Cohesion: 0.40
Nodes (4): Kompilieren, LaTeX — Projektarbeit (Ausarbeitung), Struktur, Zweck

## Knowledge Gaps
- **4 isolated node(s):** `Zweck`, `Struktur`, `Kompilieren`, `Inhaltlich offen (hängt am Projekt, nicht am Dokument)`
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LaTeX — Projektarbeit (Ausarbeitung)` connect `LaTeX — Projektarbeit (Ausarbeitung)` to `Nächste Schritte (TODO)`?**
  _High betweenness centrality (0.933) - this node is a cross-community bridge._
- **Why does `Nächste Schritte (TODO)` connect `Nächste Schritte (TODO)` to `LaTeX — Projektarbeit (Ausarbeitung)`?**
  _High betweenness centrality (0.333) - this node is a cross-community bridge._
- **What connects `Zweck`, `Struktur`, `Kompilieren` to the rest of the system?**
  _4 weakly-connected nodes found - possible documentation gaps or missing edges._