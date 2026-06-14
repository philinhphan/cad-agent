# Reproject-as-Eval — getroffene Entscheidungen

Kurzreferenz der Design-Entscheidungen für die Integration von `reproject_check`
als Per-Iteration-Signal in die Self-Refine-Loop (nur Drawing-Modus).

1. **Kein Hard-Gate** — reproject ändert `effective_score`/Accept/Champion-Wahl nicht *direkt*; Accept bleibt `Critic-Score ≥ threshold`. Aber der Critic **muss** ein failed reproject respektieren (Soft-Cap, siehe 10).
2. **Nicht bestrafen** — Zeichnung nicht parsebar (keine Views) oder alle Views global niedrige Coverage (vermutlich Orientierung/Skalierung) → `evaluated=False`, Signal weglassen statt abwerten.
3. **Routing** — Overlay-**Bilder** → Generator (nächste Iteration); Zahlen-**Digest** (Text) → Critic.
4. **Critic-Input** — Digest-Text **und** das Overlay-Composite-Bild (geändert ggü. ursprünglich „nur Text": der `sucess3`-False-Accept zeigte, dass der Critic den sichtbaren Fehler ohne Bild übersieht).
5. **Aufbereitung** — dünner Adapter in `cad_gen`; die **Scoring-Logik** von `reproject_check` (HLR, Fit, chamfer/coverage) bleibt unangetastet (nur die View-Eingabe wurde extern gemacht, siehe 9).
6. **Overlay = Lokalisierer, kein Maßband** — Overlay ist dimensionslos. Generator nutzt es nur, um die Fehlstelle zu finden; korrekte Maße immer aus der **Original-Zeichnung** lesen.
7. **Subprozess-Aufruf** — reproject läuft als Subprozess (Crash/Hang-Isolation, spiegelt `sandbox/executor.py`).
8. **Mehrere Zeichnungen** — das **erste** Drawing wird reprojiziert (reproject erwartet ein Orthographie-Sheet).
9. **View-Lokalisierung per VLM** — die CV-Heuristik `locate_views` (nur 4/9 zuverlässig) ist **entfernt**. Ein kleiner VLM-Call (`CAD_GEN_VIEW_MODEL`, „provider:model") findet die front/top/side-Boxen **einmal pro Run**; die deterministische Überlappung urteilt weiter (VLM kann nur False-Negatives erzeugen, nie einen Fake-Pass).
10. **Soft-Cap statt Hard-Gate** — bei reproject `passed=False` darf der Critic kein matches_spec / Score ≥ 8 vergeben, außer das Overlay zeigt die markierten Views als korrekt. Schließt False-Accepts (`sucess3`: zu tiefer Schnitt, 10/10) ohne Loop-Gate — robust gegen reproject-False-Negatives.
