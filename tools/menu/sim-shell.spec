action shell "Interaktive Shell im Container" \
  --rank 10 \
  --group "Werkzeuge" \
  --hint  "Wird als einzige Aktion NICHT ins Log gespiegelt — interaktives -it vertraegt die tee-Pipe nicht."

param HF_TOKEN secret "" expert "HuggingFace-Token"
