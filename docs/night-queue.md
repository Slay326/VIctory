# Nacht-Warteschlange (Night 2, ab 26.11.2026)

Stand: 26.11.2026 ~20:00, Branch `play` (79244da). Angelegt von senior 44 (Opus) für Night-2-Punkt 4.
Regeln: Night-2-Regeln des Nutzers (lokal). Fable füllt die Warteschlange nach, Opus arbeitet sie der
Reihe nach ab. Zahlenmarken: **[M]** gemessen, **[C]** per Skript berechnet bzw. aus Code abgeleitet,
**[S]** geschätzt. Alle Dauern sind [S].

Legende je Karte: **Ziel / Schritte / Abnahme / PC / Aufwärmen / Abhängig von / Dauer / Wer**.
- **Wer = Opus**: Opus darf die Karte allein abarbeiten.
- **Wer = FABLE ONLY**: nur Fable implementiert (Sicherheitsnetz, Buchführung pro Submission, alles mit
  Reihenfolge zwischen CPU, GPU und Copy-back). Opus darf dazu nur Tests vorbereiten.
- **Review Fable**: Jeder Opus-Diff an Speicher-Sync, Seitenverfolgung, Copy-back oder Buchführung braucht
  Fables Review, bevor er in `play` gemergt wird.
- Ein GPU-Job pro PC gleichzeitig. PC 1 (3070) = fps/Speicher, A-B-A. PC 2 (4070) = Bild, RT, Absturz.
- Jeder neue Test wird rot ohne Fix und grün mit Fix gezeigt; sonst zählt er nicht.
- Status: `offen` / `läuft (wer, seit)` / `fertig (Beleg)` / `blockiert (Grund)`.

Summe der Dauern (Skript, Eingaben = die [S]-Dauern der Karten): Stufe-1-Karten Q1-Q8 640 min = 10,7 h,
alle Opus-Karten 1.000 min = 16,7 h [C aus S]. Mit zwei PCs und CPU-Arbeit parallel reicht das für die Nacht.

---

## Stufe 1: Klassifizieren (messen, nichts ändern)

### Q1 Instrumentierung je Speicherregion (VICTORY_MEMCLASS=1)
- Status: blockiert (wartet auf Fable-Antwort F1 in `docs/fable-inbox.md`: Granularität, GPU-Lese-Proxy,
  CPU-Lesen nach GPU-Write)
- Ziel: Je Frame und je Seite vier Merkmale: CPU-Write, CPU-Read-nach-GPU-Write, GPU-Write, GPU-Read.
  Zusammengefasst je Region (Titel-Allokation, physische Section/View, Host-Kopie). Nur Host-Code.
- Schritte:
  1. Dichtes Merkmal-Array je 64-KiB-Seite unter 2^40 (4 Bit je Seite = 8,0 MiB [C]), Setzen per
     `fetch_or` ohne Guard-Lock. Frame-Grenze in `watch::frame_done` (watch.rs:774).
  2. CPU-Write: Guard-Handler `on_write` (watch.rs:1131, Lock bei :1147) und `Writing` (watch.rs:809) getrennt
     taggen (Titel vs. Runtime/Kernel). 4-KiB-Untermaske aus der Fault-Adresse nur, wenn F1 das will.
  3. GPU-Write: `copy_back_marks` (shadow.rs:1530, Marken-Bitmap je 64 KiB), deklarierte Bereiche
     `gpu_writes` (shadow.rs:806), Kartenkopien `card_dirty` (shadow.rs:175).
  4. GPU-Read (Proxy laut F1): Refresh-Schritte (host.rs:2492 `Step::Refresh`), Seitentabelle
     `memory::prepare` (memory.rs:567), Heap-Bereiche `Viewed::ranges` (bindless.rs:147).
  5. CPU-Read-nach-GPU-Write: nur nach Fables Entscheidung (Lese-Guards stichprobenweise oder gar nicht).
  6. Regionen: Allokationen aus `XMemVirtualAlloc` (victory-hle modules/xmem.rs:87), `XMemAllocatePhysicalPages`
     (:477), Views (`map`, :808); Host-Kopien (`Shadow::new_at`, shadow.rs:685).
  7. Ausgabe: alle 5 s eine Zeile je Klasse (Bytes, Regionen, MB Sync/Frame, ms Sync/Frame), am Laufende
     eine Tabelle je Region (CSV-artige Logzeilen, Format aus Q2).
  8. Schalter aus = kein Codepfad aktiv (keine Atomics, kein Array).
- Abnahme: `cargo test --workspace` grün bis auf die bekannten Ausnahmen (xtask-Policy lower.rs,
  `view_addresses_cover_tables_made_afresh`, Skip `heap_direct_texture_reads_mark_their_heap_index`);
  clippy -D warnings mit/ohne diagnostics; SOURCE_HASH im exe = 05856112605a1cd8 (sonst kein Lauf ohne
  neuen Cache); Tests aus Q3 grün; Fable-Review des Diffs.
- PC: Build auf PC 1 (nicht während eines Aufwärmens). Aufwärmen: **nein** (nur Host-Code).
- Abhängig von: F1. Dauer: 2,5 h. Wer: Opus (Review Fable).

### Q2 Auswertungsskript Klassifizierungstabelle
- Status: offen (Logformat erst nach F1 endgültig; Skelett mit Testdaten vorher möglich)
- Ziel: Aus run.log eine Tabelle: Region, Größe, Klasse (GPU-only / CPU-only / shared / unknown),
  Frames mit jedem Merkmal, MB und ms Sync pro Frame (Refresh, Snapshot, Kartenkopie, Copy-back),
  Anteil je Klasse an den 7,5 GB Host-Kopien [M] und den 940 MB Kartenkopien [M].
- Regeln für die Klassen (Vorschlag, Fable bestätigt in F1):
  GPU-only = GPU-Write oder GPU-Read, nie CPU-Write, nie CPU-Read-nach-GPU-Write im ganzen Lauf;
  CPU-only = CPU-Write, nie GPU-Read/GPU-Write; shared = beides; unknown = kein Merkmal oder Lücke
  (z. B. ungeschützte Seiten, Lauf abgebrochen).
- Schritte: Python (wie `pc1eval.py`), Medianbildung über Welt-Intervalle (>= 300 Draws/Frame), eine
  Zeile je Region plus Summen; Selbsttest mit synthetischem Log (je Klasse ein Fall, plus kaputtes Log).
- Abnahme: Selbsttest rot bei absichtlich falscher Klassenregel, grün mit richtiger; Ausgabe auf einem
  echten Lauf reproduzierbar (zweimal gleich).
- PC: beliebig, CPU. Aufwärmen: nein. Abhängig von: F1 (Format). Dauer: 1,5 h. Wer: Opus.

### Q3 Tests der Instrumentierung
- Status: offen (nach Q1-Schritt 1)
- Ziel: Unit-Tests ohne Spieldaten: Merkmale werden je Frame zurückgesetzt; Titel-Write vs. Runtime-Write
  getrennt; GPU-Marke setzt nur GPU-Write; Region fasst Seiten richtig zusammen (Alias-Views zählen
  zur Section, nicht doppelt); Schalter aus = keine Merkmale.
- Abnahme: jeder Test einmal rot mit absichtlich falscher Implementierung (Log zeigen), dann grün.
- PC: PC 1 oder PC 2 (CPU). Aufwärmen: nein. Abhängig von: Q1. Dauer: 1 h. Wer: Opus.

### Q4 Messlauf PC 1: Kosten und Tabelle (A-B-A)
- Status: blockiert (Q1, Q2)
- Ziel: Kosten der Instrumentierung und erste Tabelle auf der 3070.
- Schritte: exe aus Q1, Cache `play-cache`, Env wie play-Baseline (Lauf 20261126-192359: CLAMP_WINDOW=1,
  BACK_OWN_VIEW=1, BACK_THREADS=4, VIEW_RECLAIM=32, EVICT_UNWRITTEN=1, TEXTURE_HEAPS_LEAN=1,
  CARD_BUFFERS_MB=896, BVH an), A = ohne MEMCLASS, B = MEMCLASS=1, A-B-A je 480 s, `seq-pc1.ps1`
  (leert VICTORY_* zwischen den Läufen), Speicherprotokoll an. Auswertung `pc1eval.py` + Q2.
- Abnahme: B - A innerhalb der A-A-Streuung (Referenz 10,7 ms [M], Fable-Plan 1.2) oder Kosten
  benannt; kein Absturz; Tabelle vollständig (unknown-Anteil in Bytes angegeben).
- PC: **PC 1**. Aufwärmen: nein. Abhängig von: Q1, Q2. Dauer: 45 min. Wer: Opus.

### Q5 Messlauf PC 2: Tabelle bestätigen (BVH an und aus)
- Status: blockiert (Q1, Q2)
- Ziel: Gleiche Klassen auf der 4070 über die ganzen 8 min; BVH an und aus.
- Schritte: Build `targets\play`-Muster mit Q1-Commit, Cache `caches\play`, Aufgabe mit LogonType
  Interactive, PC2-NEXT belegen; 2 Läufe à 480 s (BVH an, BVH aus), gleiche Capture-Frames.
- Abnahme: Klassen je Region auf beiden PCs gleich für >= 95 % der Bytes (Schwelle laut F1); Abweichungen
  einzeln gelistet; Bild gegen play-Baseline (mittlere Differenz im Rauschen, Referenz 3,8 / ~4 % [M]).
- PC: **PC 2**. Aufwärmen: nein. Abhängig von: Q1, Q2. Dauer: 40 min. Wer: Opus.

### Q6 Tests der Invarianten (Night-2-Regeln, "Tests")
- Status: fertig (senior 47, 26.11. ~20:45). Beleg: `crates/victory-gpu/src/shadow_invariants.rs` (7 Tests
  (a)-(c), 3 `#[ignore]`-Gerüste (d) für Z2), einziger Eingriff außerhalb: 6 Zeilen `#[cfg(test)] mod invariants;`
  in shadow.rs. Rot-Beleg: 7 absichtliche Fehler in shadow.rs (lokal, nicht committet), jeder der 7 Tests mindestens
  einmal rot, Original grün (JOURNAL "senior 47"). `cargo test --workspace` (--no-fail-fast, --skip
  heap_direct_texture_reads_mark_their_heap_index): 935 ok, 2 rot (beide vorbestehend: xtask-Policy lower.rs,
  descriptor::host view_addresses_cover_tables_made_afresh), 14 ignoriert. Fable-Review der Testannahmen: Inbox F2a.
- Ziel: Synthetische Tests auf dem heutigen Code, die später den Umbau absichern:
  (a) eine von der CPU geschriebene Seite ist vor jedem GPU-Lesen synchronisiert (Refresh nach Guard-Fault);
  (b) eine von der GPU geschriebene Seite ist am Sync-Punkt für die CPU sichtbar (Copy-back markierter Seiten
  vor dem Bottom-of-pipe-Write); (c) verschachtelte Submissions mit simulierter Reihenfolge (zwei Listen,
  Titel schreibt zwischen ihnen). (d) "demotete Seite wird nie wieder GPU-only" erst, wenn es Demotion gibt:
  bis dahin nur als `#[ignore]`-Gerüst mit Spezifikation, zählt noch nicht.
- Schritte: an bestehende Tests anlehnen (`watch::tests`, `shadow::tests::writes_to_a_copy_in_gpu_memory_go_back`,
  `memory::tests::a_submission_that_missed_leaves_the_copies_at_once`); Rot-Beleg über einen Test-Schalter
  bzw. eine absichtlich ausgelassene Refresh-/Copy-back-Zeile in einem lokalen Patch (nicht committen).
- Abnahme: (a)-(c) rot mit Patch, grün ohne; Fable-Review vor Merge (berührt Sync-Semantik).
- PC: PC 1 oder PC 2 (GPU-Tests brauchen eine Karte, kurz; nicht während eines Spiel-Laufs).
  Aufwärmen: nein. Abhängig von: nichts. Dauer: 2 h. Wer: Opus (Review Fable).

### Q7 Test für das Lauf-Skript (Env-Leck, Timeouts, Parsing)
- Status: fertig (senior 47, 26.11. ~20:45). Beleg lokal (nicht im Repo): Agenten-Ordner `scripts/tests/`
  (`test_runscripts.py`, Fixture aus Lauf 192359, Attrappe `stub/`, `redproof.py` -> `redproof.txt`). 10 Tests,
  9 grün + 1 erwarteter Fehlschlag = Befund: n1run.ps1 wartet im finally per Prozessname (`Get-Process victory`)
  und ohne Limit auf jede victory.exe (Entscheidung des Nutzers offen, JOURNAL "senior 47").
- Ziel: Die lokalen Lauf-Skripte (`seq-pc1.ps1`, `n1run.ps1`, `pc1eval.py`, `memrep.py`, PC-2-Pendants)
  sind getestet: (a) kein VICTORY_* sickert in den Folgelauf (Vorfall 150437, JOURNAL 16:40); (b) harter
  Timeout greift (8 min + Rand) und beendet nach ExecutablePath, nicht nach Prozessname; (c) Parser liefert
  auf einem Referenz-Log dieselben Mediane wie heute (192359: 343,9 ms = 112,4 + 182,6 + 46,5 [M]).
- Schritte: Pester- bzw. Python-Tests mit einer Attrappe statt victory.exe (schläft, gibt Env aus);
  Referenz-Log-Ausschnitt lokal. Die Skripte liegen nicht im Repo: Tests ebenfalls lokal, nichts committen.
- Abnahme: jeder Test rot gegen eine absichtlich kaputte Skriptkopie (Env nicht geleert, Timeout aus,
  Regex falsch), grün gegen das Original.
- PC: PC 1 (CPU, keine GPU). Aufwärmen: nein. Abhängig von: nichts. Dauer: 1,5 h. Wer: Opus.

### Q8 Daten für die nächste Fable-Frage (Stufe 2)
- Status: blockiert (Q4, Q5)
- Ziel: Kompaktes Paket für "welcher Block zuerst" (Stufe 3): die 5 größten GPU-only-Regionen mit Größe,
  Sync-MB/ms pro Frame, Herkunft (Allokationsart), Stabilität über den Lauf, Abweichung PC 1/PC 2.
- Schritte: Skript Q2 auf Q4/Q5 anwenden, Ausgabe ins JOURNAL, Frage in `docs/fable-inbox.md` (F2).
- Abnahme: jede Zahl markiert und mit Lauf-ID; Skript gespeichert. PC: CPU. Aufwärmen: nein.
  Dauer: 45 min. Wer: Opus.

---

## Nebenkarten (nur in freien GPU-Slots, nie vor Stufe-1-Karten)

### R1 REA: CPU-Code der Spiel-Binary, der GPU-Puffer liest oder schreibt
- Status: offen
- Ziel: Statisch finden, welche Titel-Funktionen Speicher aus `XMemVirtualAlloc`/`XMemAllocatePhysicalPages`
  (GPU-Pools) per CPU lesen bzw. schreiben, als Erklärung für Regionen der Klasse shared/unknown.
- Schritte: nur in PC-Pausen (kein Spiel-Lauf, kein Aufwärmen auf demselben PC); vorher freien RAM prüfen
  (Abbruch unter 8 GB frei [S]); Aufrufer der Importe für die XMem-Funktionen, Weitergabe der Rückgabezeiger,
  Lese-/Schreibzugriffe darauf; Ergebnisse nur lokal (Agenten-Ordner außerhalb des Repos), Binary danach schließen.
- Abnahme: Liste Funktionsadresse -> Zugriffsart -> Allokationsart, mit Abdeckung/Grenzen der Analyse.
  **Nichts davon committen** (Asset-Policy: kein dekompilierter Code, keine Strings der Binary im Repo).
- PC: PC 1 oder PC 2, CPU/RAM-lastig. Aufwärmen: nein. Abhängig von: nichts. Dauer: 1,5 h. Wer: Opus.

### R2 VRAM PC 1: play über Budget
- Status: Analyse fertig (senior 45, JOURNAL ~20:00, unverified, for Fable): Schalter greifen; Karte +581 MB
  gegenüber s32 kommt aus Host-Texturen aus Titelspeicher (+858 MB, 1.886 -> 2.605 verschiedene Texturen),
  schon in r25 so (Inhalt durch main-Shader-Fixes, kein Leck); Budget 7.239 statt 7.417 nicht durch
  Fremdprozesse. Offen nur noch Schritt 5 (unten). Schritte 1-4 entfallen.
- Ziel: Erklären, warum die 3070 in play 8.046 von 7.239 MB belegt [M, Lauf 192359], obwohl dieselben
  Schalter (EVICT_UNWRITTEN + TEXTURE_HEAPS_LEAN + CARD_BUFFERS_MB=896, im meta.txt bestätigt) vorher
  7.477 von 7.417 MB ergaben [M, Lauf 20261126-161102].
- Schritte: (1) Budget-Verlauf (DXGI) beider Läufe per Skript vergleichen (7.239 vs. 7.417: andere Prozesse
  auf der Karte? `nvidia-smi --query-compute-apps`); (2) "VRAM by kind" beider Läufe Spalte für Spalte
  (Host-Texturen 5.877 vs. ~5.000 [M]); (3) Unterschied r25/play vs. s32 im Texturpfad (Diff lesen);
  (4) nur wenn (1)-(3) nichts erklären: ein Lauf play mit denselben Schaltern bei geschlossenen
  GPU-Fremdprozessen.
  (5) Fable entscheidet zwischen senior 45s Optionen: (a) Bisect r25 vs. s32 nur für den Textur-Zuwachs
  (Aufwärmen je Schritt), (b) Deckel bzw. früheres Abgeben ungenutzter Titel-Texturen hinter Schalter,
  (c) A-B play mit/ohne NVIDIA-Overlay zum Budget. Danach ggf. eine Karte für Opus.
- Abnahme: Ursache mit Zahlen belegt oder als "unverified, for Fable" markiert; keine neue Design-Entscheidung.
- PC: **PC 1** (nur für (c)). Aufwärmen: nein (außer (a)). Abhängig von: Fable. Dauer: 30 min. Wer: Opus.

### R3 RT: gezielter Refresh nicht überwachter Builder-Eingaben
- Status: offen (Branch `test/bvh31-blimit`, nicht `play`)
- Ziel: Die Builder-Eingaben aus nicht write-watched Titelspeicher (54 % der Arena-Seiten [M, Lauf 185839])
  vor jeder Builder-Submission neu laden, aber nur die Seiten, auf die seine Root-Views bzw. Eingabelisten
  zeigen, statt reihum (REFRESH_UNWATCHED=512 kostete 0,6 fps, Vorbereitung 1.155 ms [M]).
- Schritte: auf c8e6822 aufsetzen; Auswahl = Schnitt aus "nicht watched" und "vom Builder in den letzten
  N Frames gelesen/verfehlt"; Zähler "unwatched builder inputs sent again N (MB)"; synthetischer Test:
  unwatched Seite ändert sich, Builder sieht neuen Stand.
- Abnahme: Anteil "Kindbox schließt Ursprung ein" sinkt deutlich unter 46-70 % [M]; Vorbereitung
  < +100 ms gegenüber ohne; 8 min ohne Device-Removed; Fable-Review (Seitenverfolgung).
- PC: **PC 2**. Aufwärmen: nein (Host-Code). Abhängig von: nichts. Dauer: 2 h. Wer: Opus (Review Fable).

### R4 Haar-Artefakte (Dump)
- Status: offen
- Ziel: Erste Stelle finden, an der das Haar kaputt geht (Hypothese finisher 7: Haupt-Tiefe vom
  Alpha-Prepass überschrieben, Depth-Bounds; unbestätigt auf play).
- Schritte: PC 2, play-exe, `VICTORY_DUMP_TARGETS`/`VICTORY_DUMP_WRITERS` auf eine Welt-Szene mit Ped
  in Nahaufnahme, Tiefe vor/nach dem Alpha-Prepass vergleichen; Captures lokal.
- Abnahme: erster fehlerhafter Pass benannt (Programm-Hash, Ziel), Befund im JOURNAL, keine Captures im Repo.
- PC: **PC 2**. Aufwärmen: nein. Abhängig von: nichts. Dauer: 1 h. Wer: Opus.

### R5 Pinnwand / Schatten treppig
- Status: offen
- Ziel: Ursache der treppigen Pinnwand bzw. Bett-Schatten (bleibt nach dem FinalGather-Fix, auch BVH aus).
- Schritte: PC 2, `VICTORY_DUMP_AFTER` (+ `VICTORY_DUMP_AFTER_MAX`) auf die Schatten-Passes
  (Kaskaden, Shadowmask) im Frame mit sichtbarer Treppe; Auflösung/Format der Schattenziele und
  Sampler-Vergleichsmodus prüfen.
- Abnahme: erster Pass mit Treppe benannt; Hypothese als "unverified, for Fable" ins Inbox, wenn sie eine
  Übersetzer-Änderung verlangt.
- PC: **PC 2**. Aufwärmen: nein (Dump). Abhängig von: nichts. Dauer: 1 h. Wer: Opus.

### R6 RT: upstream #2098 (Adresse der Acceleration-Structure-View) auf rt/play testen
- Status: offen (Prüfung senior 49; Kandidaten-Rangliste im JOURNAL unter "senior 49")
- Ziel: Der RT-Leser f607780a nimmt die BVH-Adresse aus den View-Adressen (Dwords 0-1). Für eine
  SRV der Dimension 11 (Acceleration Structure, ohne Ressource) schreibt `write_address` heute 0
  [C, descriptor/host.rs auf rt/play 2e852e6; Diag-Log: dim 11, location 0xd5d3ef800, eingetragen 0x0, JOURNAL Z. 210].
  Der Leser läuft also über Adresse 0 statt über das TLAS. #2098 trägt die Location (Dwords 4-5) ein.
  Erklärt NICHT die stillen Nullen im Builder (der bindet keine Dim-11-View [M, Lauf 20261124-230706:
  nur f607780a]), betrifft aber direkt das RT-Bild.
- Schritte: Branch `test/rtplay-2098` = `fork/rt/play` (2e852e6) + `git cherry-pick -x 543d868`
  (Autor moe-miau bleibt; Commit-Nachricht nennt "upstream PR #2098, 543d868"); Konflikte: keine
  [C, merge-tree]. Nur victory-gpu -> SOURCE_HASH gleich, KEIN Aufwärmen (Cache von rt/play).
  Test `descriptor::host::tests::an_acceleration_structures_view_address_is_its_location` rot ohne /
  grün mit Fix zeigen, dann `cargo test --workspace`. Push fork `test/rtplay-2098`, PC 2 zieht.
- Messung (PC 2, 180 s, gleiche Einstellung und Capture-Frames wie rt/play-Kontrolle, A-B-A):
  (1) Bild: RT-Schatten/Irradiance auf den festen Capture-Frames gegen Kontrolle (Bett-Maske, Luminanz);
  (2) Kontrolle Builder-Seite: "BVH GT reader" gültige Bottom-Level 10-20 % und Ursprungs-Anteil 46-70 %
  sollten GLEICH bleiben (sonst Befund melden); (3) Arena-Zeile "pages asked/refused" (der Leser liest
  jetzt die TLAS-Seiten 0xd...); (4) GPU-Zeit von f607780a und Device-Removed (Risiko: Durchlauf über
  echte, teils ungültige BVHs bis 65.535 Schritte je Wave).
- Abnahme: sichtbarer Unterschied im RT-Bild mit Captures belegt ODER "keine Wirkung" mit Zahlen;
  kein neuer Device-Removed in 180 s; Builder-Metriken unverändert dokumentiert.
- PC: **PC 2**. Aufwärmen: nein (Host-Code). Abhängig von: nichts. Dauer: 45 min. Wer: Opus.
- Nicht testen (senior 49, ohne erwartete Wirkung): #2467 (Root-Pfad; f607780a nimmt den Roh-Pfad,
  der die hohen Bits schon hält; #2501 an denselben Zeilen gab byte-gleiche Pipelines [M, 04:25]),
  #1794, #3316 (alle 64 Leser übersetzen schon), #1903 (ULP), #2853 (Konflikt, CI rot), #1885 (Proton).

---

## Werkzeuge und parallele Streams (senior 55, 26.11. ~21:30; Hintergrund `docs/dev-efficiency.md`)

### Q9 Laufzeit-Schalter für Host-Optionen (A-B im selben Spiellauf)
- Status: offen (Rangfolge wartet auf Fable F4; die Karte ist unabhängig davon baubar)
- Ziel: Host-Optionen im laufenden Spiel umschalten, damit ein A-B in einem Lauf entsteht (gleiche Szene,
  gleicher Speicherzustand, ein Laden statt zwei). Nur Host-Optionen (z. B. `VICTORY_BACK_THREADS`, Kopierwege,
  `VICTORY_CLAMP_WINDOW`); nichts, was beim Start Ressourcen, Heaps, Caches oder Pipelines anlegt.
- Schritte:
  1. Neue Funktion neben `switch::on` (victory-gpu `switch.rs`): Wert aus einer Steuerdatei
     (`VICTORY_SWITCH_FILE=<pfad>`), gelesen höchstens einmal je Frame an der Frame-Grenze, sonst Env-Wert.
  2. Zeitplan-Modus: `VICTORY_SWITCH_EVERY=30` wechselt A/B alle 30 s selbst (Datei optional), damit kein
     externer Taktgeber nötig ist.
  3. Nur Schalter umstellen, die heute per `OnceLock` gelesen werden und die laut Code-Durchsicht ohne
     Neuanlage wechseln dürfen; Liste in der Karte nachtragen. Wechsel nur an der Frame-Grenze, nie mitten in einer
     Submission.
  4. Log: bei jedem Wechsel eine Zeile `switch interval <n> <A|B> <schalter=wert,...> frame=<f>`.
  5. `pc1eval.py` (lokal): Intervalle je A/B auswerten (Median Vorbereitung / GPU / Zurück, fps, Streuung),
     erstes Intervall nach jedem Wechsel als Einschwingen verwerfen.
- Abnahme: Unit-Test für den Wechsel (Datei/Zeitplan -> Wert an der Frame-Grenze, kein Wechsel mitten im Frame),
  rot ohne / grün mit Umsetzung gezeigt; Log markiert jedes Intervall; pc1eval wertet je Intervall aus (Test mit
  Fixture-Log wie Q7); `cargo test --workspace` grün bis auf bekannte Ausnahmen. Erster Spiellauf erst nach den
  Offline-Tests: PC 1, 180 s, A = B (gleicher Wert) als Leertest, Unterschied muss im Rauschen liegen.
- PC: Bau/Tests PC 1 (CPU), Beweislauf PC 1. Aufwärmen: nein (Host-Code). Abhängig von: nichts.
  Dauer: 4 h [S]. Wer: Opus. Review Fable, falls ein umgeschalteter Schalter Copy-back oder Sync betrifft.

### Q10 Clean-Room: Guard-Faults billiger
- Status: offen. **Nur für einen Agenten, der weder xcl- noch console_insn-Code gelesen hat (nicht senior 50/51).**
  Die Beschreibung unten ist die einzige Grundlage; kein Bezug auf fremden Code.
- Ausgang: ~1.128 Guard-Faults pro Frame à ~64 µs [M, play-Baseline] = 72,2 ms/Frame seriell [C, devstats.py].
- Ziel: Guard-Faults auf überwachten Seiten billiger machen, ohne die Seitenverfolgung zu verfälschen.
  1. **Einzelne Schreibzugriffe im Fault-Handler ausführen**: Wenn der Titel auf eine überwachte Seite schreibt,
     den einen auslösenden Befehl dekodieren und seine Wirkung (Ziel, Wert, Breite) im Handler selbst ausführen,
     die Seite als vom CPU beschrieben vermerken und hinter dem Befehl weitermachen, statt die Seite freizugeben,
     den Block zu kopieren und wieder zu schützen. Unbekannte Befehle gehen den heutigen Weg (sicherer Rückfall).
  2. **Gleiche Werte nicht schreiben**: An heißen Schreibstellen, die meist einen schon vorhandenen Wert
     schreiben (z. B. Flags setzen, die schon gesetzt sind), zuerst lesen und nur bei Änderung schreiben und
     markieren. Die Stellen findet man zur Laufzeit an ihren Faults (Zähler je Befehlsadresse, Anteil
     "Wert unverändert").
- Schritte: (a) Zähler je Fault-Adresse und Anteil unveränderter Werte (nur messen); (b) Dekoder/Emulator für die
  häufigsten Befehlsformen mit Tabellen-Tests gegen echte CPU-Ausführung; (c) Handler-Weg hinter Schalter
  `VICTORY_GUARD_EMULATE=1` (Standard aus); (d) Offline-Tests zuerst, Spiellauf nur als Beweis.
- Abnahme: Tests ohne Spiel: emulierter Befehl == echte Ausführung (Register, Speicher, Flags) für jede
  unterstützte Form, unbekannte Form -> alter Weg; Seite ist nach emuliertem Write als CPU-beschrieben markiert und
  wird vor dem nächsten GPU-Lesen synchronisiert (Invariante wie Q6); jeder Test rot ohne / grün mit Umsetzung.
  Dann PC 1 A-B-A (Faults/Frame, µs/Fault, fps, Streuung), PC 2 Bildvergleich im Rauschen.
  **Danach Fable-Review**, weil es die Seitenverfolgung betrifft; kein Merge nach play ohne Review.
- PC: Bau/Tests CPU, Beweis PC 1 + PC 2. Aufwärmen: nein (Host-Code). Abhängig von: nichts.
  Dauer: 6 h [S]. Wer: Opus (Clean-Room-Agent), Review Fable.

### S1-S4 Parallele Streams (Owner laut JOURNAL-Tabelle "STREAMS")
- **S1 BVH-Capture + Harness, PC 2**: in Arbeit (senior 52).
- **S2 BVH-Hypothesen, PC 2 offline**: offen bis das Capture aus S1 steht; dann 2-3 Agenten, je eine Hypothese,
  Fable urteilt gebündelt.
- **S3 Kostenmodell + Mikro-Benchmarks, PC 1**: in Arbeit (senior 53); prüft auch GPU-Replay (Werkzeug c).
- **S4 Stufe-1-Vorbereitung, PC 1, offline bis 23:00**: in Arbeit (senior 54); legt das Ereignisformat für
  Host-Record/Replay an (Werkzeug b), damit Stufe 3 gegen Captures getestet wird.

---

## Nur Fable (Opus bereitet höchstens Tests/Daten vor)

### Z1 Stufe 2: Design `docs/memory-model.md` und Go/No-Go je Stufe — FABLE ONLY
- Abhängig von: Q4, Q5, Q8. Opus: Daten (Q8).

### Z2 Stufe 3: Sicherheitsnetz (Seite erkennen, synchronisieren, demoten) — FABLE ONLY
- Abhängig von: Z1. Opus: Test-Gerüste (Q6 d), Demotion-Zähler im Auswerteskript.

### Z3 Buchführung pro Submission (Grundlage für Flights) — FABLE ONLY
- Abhängig von: Z1 und bewiesener Stufe 3.

### Z4 Review aller Opus-Diffs an Sync/Seitenverfolgung/Copy-back/Buchführung — FABLE ONLY
- Betrifft Q1, Q3 (falls Semantik), Q6, R3.

---

## Reihenfolge-Vorschlag für die Nacht

| Zeit [S] | PC 1 (3070) | PC 2 (4070) | CPU (ohne GPU) |
|---|---|---|---|
| ab 23:00 | Q6 Tests (kurz GPU) | R3 RT-Refresh | Fable: F1; Opus: Q7, Q2-Skelett |
| nach F1 | Q1 bauen, Q3 | R4 Haar-Dump | Q2 |
| danach | Q4 A-B-A | R5 Schatten-Dump | Q2 fertig |
| danach | R2 (c), falls Fable es will | Q5 BVH an/aus | Q8, F2 schreiben |
| Pausen | R1 (wenn kein Lauf) | R1 (wenn kein Lauf) | - |
