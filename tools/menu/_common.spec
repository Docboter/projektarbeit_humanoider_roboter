# _common.spec — Parameter, die sich mehrere Aktionen teilen.
#
# Wird VOR jeder Aktions-Spec geladen; direkt danach _common-<prefix>.spec (also
# _common-sim.spec bzw. _common-train.spec). Hier steht nur, was BEIDE Welten teilen. Eine Aktion, die einen dieser Parameter anders
# einstuft, deklariert ihn einfach erneut — die spaetere Fassung gewinnt und bestimmt
# auch die Reihenfolge. So steht HF_TOKEN bei 'eval' vorn und ist bei 'view' auf
# 'expert' heruntergestuft (dort braucht es gar keine Gewichte).

group "Zugang"
param HF_TOKEN secret "" basic \
  "HuggingFace-Token" \
  "Wird fuer den Checkpoint-Download gebraucht. Dauerhaft besser in .env.local ablegen (: \"\${HF_TOKEN:=hf_...}\") — dann fragt das Menue hier nicht mehr und der Wert landet nie in einer Recall-Datei." \
  --default-from "Geheimnis - hat per Definition keinen Default"
