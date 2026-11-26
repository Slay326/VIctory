# Entwicklungseffizienz (Night 2, Stand 26.11.2026 ~21:30)

Angelegt von senior 55 (Opus). Zahlenmarken: **[M]** gemessen/gezählt aus Logs, **[C]** per Skript berechnet,
**[H]** Stichwort-Heuristik des Skripts (Liste zur Prüfung in der Skriptausgabe), **[S]** geschätzt,
**[LB]** Untergrenze (Quellen unvollständig). Skript: `scripts/devstats.py` im Agenten-Ordner (lokal, nicht
im Repo), Aufruf `devstats.py --end "2026-11-26 21:00" --hours 48`; Ausgabe steht im JOURNAL unter "senior 55".
Alle Bewertungen in Abschnitt 3 und 4 sind **Opus-Schätzung, unverified, for Fable**. Das Urteil fällt Fable (Inbox F4).

## 1. Letzte 48 h (24.11. 21:00 bis 26.11. 21:00)

| | PC 1 (3070) | PC 2 (4070) | Summe |
|---|---|---|---|
| Spielläufe [M] | 237 | 90 | 327 [C] |
| Laufzeit inkl. Laden [C] | 19,6 h | 8,5 h | 28,1 h |
| bis zum Zeitlimit gelaufen [M] | 159 | 63 | |
| Absturz (Device-Removed / Zugriffsverletzung / PC 2 früh + crashes.log) [M] | 27 / 32 | 24 | 83 = 25,4 % [C] |
| abgebrochen ohne meta.txt (Reaper, Skript gekillt) [M] | 14 | 3 | 17 |
| kürzer als 180 s [C] | 62 | 23 | 85 = 26,0 % |
| Aufwärmen mit Logdatei [M, LB] | 23 / 5,5 h | 10 / 2,1 h | 7,6 h |
| GPU-Wanduhrzeit Läufe + Aufwärmen [C, LB] | | | 35,7 h |

- JOURNAL-Einträge im Fenster: 137 [M]. Davon nach Stichwort-Heuristik [H]: 16 Hypothese bestätigt / Ursache
  gefunden, 13 Hypothese widerlegt, 9 Änderung mit gemessenem Gewinn, 7 abgebrochen bzw. Überschneidung,
  92 Organisation / Lesen / Bauen. Getestete Hypothesen [H]: 38, also 0,94 GPU-Stunden je Hypothese [C aus H].
  Die Heuristik verwechselt einzelne Einträge (z. B. "Gewinn, aber Ziel NICHT erreicht" zählt als widerlegt).
- Bewiesen fertige Änderungen (Handliste Opus, gezählt): neue Basis d17 (GOOD), neue Basis s20, play-Baseline
  (r25 + upstream), #1906 upstream gemergt, VIEW_RECLAIM-Leckfix (A-B-A beide PCs), 2 Draft-PRs #3197/#3199
  = 7. Commits auf play, die nicht in upstream main sind, seit Fensterbeginn: 48, davon 43 kein Doku-Commit [M].
- Wiederholungen wegen Werkzeugfehlern (aus dem JOURNAL gezählt, Opus):
  - Speicher-Reaper der Agenten-Shell: mindestens 5 Läufe (Morgenbericht), dazu 040354, 051528, Teil-Lauf 094542
    und ein Aufwärmen (n1-bundle). Behoben (Reaper abgeschaltet).
  - Env-Leck in seq-pc1: 1 kontaminierter Lauf (150437, EVICT_UNWRITTEN geerbt), A-Reihe neu gelaufen (152927,
    154540); die Folgerung "never written 0" war dadurch zeitweise falsch. Behoben, Test in Q7.
  - Geteilte Pipeline-Bibliothek: 2 Aufwärmen gleichzeitig auf derselben Library (senior 13). Regel seitdem:
    eigener `VICTORY_CACHE_DIR` je SOURCE_HASH.
  - BVH aus ohne Schleifengrenze / fehlender Skip: 154843 -> neu als 155105 mit `VICTORY_BVH_LOOP_LIMIT`;
    201856 (s46-off) Device-Removed im Laden, Pipeline e30970f1 nicht in der Skip-Liste.
  - GPU-Lock-Lücke: 1 Lauf (main24, 08:49) mit Bau im Hintergrund, Messung verunreinigt.
- Lesart (**Opus-Schätzung, unverified, for Fable**): Rund ein Viertel der Läufe endet vor der Zeit, jede
  Hypothese kostet im Mittel knapp eine GPU-Stunde. Der Hebel ist nicht schnelleres Bauen, sondern weniger
  Spielläufe je Entscheidung: mehr Antworten aus einem Lauf (Abschnitt 3 d) und Antworten ganz ohne Lauf (b, a).

## 2. Was wir schon aufzeichnen können (play, `VICTORY_*` in crates/ und apps/)

| Werkzeug | Schalter | Was es liefert |
|---|---|---|
| Bild-Captures | `VICTORY_CAPTURE`, `_FRAMES`, `_WORLD` | PNG je Frame, Weltframes relativ zum Welteintritt |
| Render-Target-Dumps | `VICTORY_DUMP_TARGETS`, `_AFTER`, `_AFTER_MAX`, `_WRITERS` | Ziele nach bestimmten Draws/Dispatches, Schreiber je Ziel |
| Absturz | `VICTORY_CRASH_SAVE`, `VICTORY_D3D12_DRED`, `_DEBUG`, `_GBV`, `VICTORY_STACKS` | Titelbereiche bei Absturz, DRED-Breadcrumbs, Stacks zu Zeitpunkten |
| Zeit | `VICTORY_GPU_TIMING`, fps-Zeile (Vorbereitung / GPU / Zurück) | GPU-Zeit je Pipeline, Frame-Aufschlüsselung je 5 s |
| Speicher | copystat (Kopien je Frame nach Art, Guard-Lock-Zeit, Snapshots), VRAM by kind, `VICTORY_COPY_MODE` | MB/ms je Kopierweg; `plain`/`none` nur für Zeitmessung |
| Titelzustand | `VICTORY_WATCH`, `_PEEK`, `_POKE` | Titel-RVA beobachten bzw. setzen |
| BVH | GT-Validator `VICTORY_BVH_GT*` (nur rt/play) | gültige Bottom-Level, Ursprungs-Anteil |
| Tests ohne Spiel | `shadow_invariants.rs` (senior 47, simulierter Titelspeicher, 7 Tests, rot belegt), Lauf-Skript-Tests (Q7) | Sync-Invarianten, Env-Leck, Timeouts, Parsing |

Was fehlt (Opus): (1) kein Aufzeichnen von Submissions, Seitenzugriffen und Sync-Punkten in einer Datei, die man
offline abspielen kann; (2) keine Umschaltung von Host-Optionen im laufenden Spiel (alle Schalter werden einmal
gelesen, `OnceLock` + `switch::on`), jedes A-B braucht zwei Läufe inkl. Laden; (3) kein Kostenmodell, das aus
Zählern ms vorhersagt; (4) Bildvergleich und Auswertung laufen über verstreute Skripte (pc1eval, memrep,
cmp-s41, eval46) ohne gemeinsames Format; (5) Aufwärm-Dauern werden nicht einheitlich protokolliert ([LB] oben).

## 3. Machbarkeit und Aufwand (alles **Opus-Schätzung, unverified, for Fable**)

| | Werkzeug | Machbarkeit | Aufwand [S] | Nutzen für den 23:00-Plan |
|---|---|---|---|---|
| a | Kostenmodell aus Zählern + Mikro-Benchmarks (Stream 3, senior 53) | hoch: Zähler gibt es (copystat, Guard-Faults 1.128/Frame à ~64 µs [M] = 72,2 ms/Frame seriell [C]) | 1 Tag bis brauchbar, ±30 % | Stufe 3 vorab bewerten: welcher Block wie viele ms spart |
| b | Host-Record/Replay von Submissions, Seitenzugriffen, Sync-Punkten | mittel-hoch: Ereignisse entstehen an wenigen Stellen (Submit, Guard-Handler, Copy-back); Replay gegen `Shadow` ohne GPU wie in den Invarianten-Tests | Format + Recorder 0,5 Tag (Stream 4), Replayer 1 Tag | Stufe 3 offline: Klassifizierung, Demotion-Netz und Invarianten gegen echte Zugriffsfolgen, bevor das Spiel läuft |
| c | GPU-Replay eines ganzen Frames mit geänderten Shadern (PIX/RenderDoc, Stream 3 prüft) | mittel-niedrig: Titelspeicher, Seitentabelle und übersetzte Pipelines müssen vollständig mit; PIX-Capture auf dem Host-D3D12 geht, Shader-Austausch darin nur für DXIL | 1-3 Tage | BVH/Bildfehler ohne Laden; für das Speichermodell wenig |
| d | Laufzeit-Schalter für Host-Optionen (Datei, Wechsel alle 30 s) | hoch für reine Host-Optionen (Kopierwege, Threads, Fenster); nicht für alles, was beim Start Ressourcen anlegt | 0,5 Tag (Karte Q9) | A-B im selben Lauf: halbiert Läufe, gleiche Szene, gleicher Speicherzustand |
| e | Statische Analyse der Spieldatei nach Schreibstellen | mittel: REA kann Funktionen finden, aber Zeiger auf GPU-Speicher sind indirekt; zur Laufzeit an Faults finden ist billiger | 1-2 Tage, Treffer unsicher | Kandidatenliste für "GPU-only", ersetzt Stufe 1 nicht |

## 4. Rangfolge (Vorschlag Opus, unverified, for Fable)

1. **b Record/Replay (Host)**: Stream 4 legt das Ereignisformat an (Submission-ID, Seitenbereich, Zugriffsart
   CPU-W/CPU-R/GPU-W/GPU-R, Sync-Punkt). Stufe 1 (Q1) schreibt dieselben Ereignisse in eine Capture-Datei; Stufe 3
   (Z2, Sicherheitsnetz) wird gegen diese Captures offline getestet, das Spiel läuft nur noch als Beweis.
2. **d Laufzeit-Schalter**: billig, sofort wirksam für jedes A-B der Nacht (Q4, Stufe 3 A-B-A).
3. **a Kostenmodell**: läuft schon (Stream 3); speist sich aus denselben Zählern wie b.
4. **c GPU-Replay**: erst nach dem Befund von Stream 3, eher für BVH/Bild.
5. **e statische Analyse**: zuletzt; die Laufzeitdaten (Faults, Stufe 1) sind billiger und genauer.

Ablauf ab 23:00: Q1 (Stufe 1) zeichnet auf -> Capture-Datei (Format Stream 4) -> Replay-Tests für Stufe 3 und das
Kostenmodell (a) -> ein Spiellauf mit Laufzeit-Schalter (d) als A-B im selben Lauf -> Beweisreihe nach den Regeln.
