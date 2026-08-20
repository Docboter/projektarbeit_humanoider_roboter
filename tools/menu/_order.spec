# TL;DR: Reihenfolge der Aktionsgruppen im Sim-Menue: bildet die Diagnosekette ab.
# _order.spec — Reihenfolge der Aktionsliste.
#
# Die Specs werden alphabetisch eingelesen; ohne diese Datei stuende die Liste in
# Dateinamen-Reihenfolge. Sie soll aber die KETTE abbilden, an der Einsteiger
# haengenbleiben:  preflight -> setup -> cams -> gap -> eval -> layout -> render -> rl.
group_order \
  "Vorbereiten" \
  "Ansehen" \
  "Messen" \
  "Co-Training vorbereiten" \
  "Trainieren" \
  "Beschleunigen" \
  "Werkzeuge"
