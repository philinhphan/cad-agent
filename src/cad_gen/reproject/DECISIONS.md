# Reproject-as-Eval — getroffene Entscheidungen

Kurzreferenz der Design-Entscheidungen für die Integration von `reproject_check`
als Per-Iteration-Signal in die Self-Refine-Loop (nur Drawing-Modus).

1. **Advisory** — reproject ändert `effective_score`, Accept-Logik und Champion-Wahl **nicht**. Der Score bleibt = Critic-Score.
2. **Nicht bestrafen** — Zeichnung nicht parsebar (keine Views) oder alle Views global niedrige Coverage (vermutlich Orientierung/Skalierung) → `evaluated=False`, Signal weglassen statt abwerten.
3. **Routing** — Overlay-**Bilder** → Generator (nächste Iteration); Zahlen-**Digest** (Text) → Critic.
4. **Critic-Input** — nur Digest-Text, **kein** Overlay-Bild (hält advisory sauber, spart Token).
5. **Aufbereitung** — dünner Adapter in `cad_gen`; die **Scoring-Logik** von `reproject_check` (HLR, Fit, chamfer/coverage) bleibt unangetastet (nur die View-Eingabe wurde extern gemacht, siehe 9).
6. **Overlay = Lokalisierer, kein Maßband** — Overlay ist dimensionslos. Generator nutzt es nur, um die Fehlstelle zu finden; korrekte Maße immer aus der **Original-Zeichnung** lesen.
7. **Subprozess-Aufruf** — reproject läuft als Subprozess (Crash/Hang-Isolation, spiegelt `sandbox/executor.py`).
8. **Mehrere Zeichnungen** — das **erste** Drawing wird reprojiziert (reproject erwartet ein Orthographie-Sheet).
9. **View-Lokalisierung per VLM** — die CV-Heuristik `locate_views` (nur 4/9 zuverlässig) ist **entfernt**. Ein kleiner VLM-Call (`CAD_GEN_VIEW_MODEL`, „provider:model") findet die front/top/side-Boxen **einmal pro Run**; die deterministische Überlappung urteilt weiter (VLM kann nur False-Negatives erzeugen, nie einen Fake-Pass).
