# Fable-Inbox (Night 2)

Offene Fragen an Fable, älteste zuerst. Je Frage: Frage, vorbereitete Daten, was blockiert ist.
Opus-Schlüsse sind als **unverified, for Fable** markiert. Zahlenmarken: **[M]** gemessen (Log-Zähler),
**[C]** per Skript berechnet bzw. aus Code abgeleitet, **[S]** geschätzt.
Skript für alle [M]/[C]-Zahlen: `memclass_inputs.py` (lokal im Agenten-Ordner `play/`, Ausgabe
`memclass-inputs.txt`; Mediane der Welt-Intervalle >= 300 Draws/Frame).

---

## F1 (für 23:00): Wie instrumentieren wir Stufe 1, damit die Klassifizierung belastbar ist?

Status: offen. Blockiert: Queue-Karten Q1 (Instrumentierung), Q2 (Tabellenformat), Q4/Q5 (Messläufe), Q8.

### Fragen (bitte je kurz entscheiden)
1. **Granularität**: Klassifizieren je 64 KiB (heutige Einheit von Guards und GPU-Marken) oder 4 KiB?
   Vorschlag Opus (unverified, for Fable): Merkmale je 64 KiB; für CPU-Writes zusätzlich eine 4-KiB-Untermaske
   aus der Fault-Adresse (ohne Zusatz-Systemaufruf); GPU-Writes bleiben 64 KiB, weil feinere Marken eine
   Shader-Änderung (Aufwärmen aller Shader) wären.
2. **GPU-Read**: Reicht ein Host-Proxy "von einer Submission erreichbar" (Refresh angefordert, Seitentabelle,
   Heap-View-Bereich)? Er überschätzt Lesen (sicher für die Klasse, aber weniger GPU-only), oder brauchen wir
   echte Lesemarken in Shadern (Aufwärmen)?
3. **CPU-Read nach GPU-Write**: heute nicht beobachtbar. Optionen: (a) Lese-Guards (PAGE_NOACCESS) auf
   GPU-geschriebene Seiten nach dem Copy-back, nur stichprobenweise (jeder N-te Frame / Teilmenge);
   (b) nicht in Stufe 1 messen und das Stufe-3-Sicherheitsnetz die Fehlklassifizierung fangen lassen;
   (c) anderes. Wenn (a): welche Stichprobe, und wie mit Kernel-Lesezugriffen (Systemaufruf liest aus
   geschützter Seite -> Fehlschlag) umgehen?
4. **Runtime-eigene Writes** (`Writing`, unsere Copy-backs, DirectStorage) als eigenes Merkmal getrennt von
   Titel-CPU-Writes? Vorschlag: ja, sonst wird jede GPU-geschriebene Seite durch unseren Copy-back "shared".
5. **Abnahme "belastbar"**: Vorschlag (unverified): unknown < 5 % der Bytes; Klasse je Region über die
   letzten 5 min stabil; PC 1 und PC 2 gleich für >= 95 % der Bytes; Instrumentierungskosten innerhalb der
   A-A-Streuung 10,7 ms [M, perf-plan 1.2]. Passt das?
6. **Messpunkte**: Sind die Stellen unten vollständig, oder fehlt ein Weg, auf dem CPU oder GPU Titelspeicher
   berührt, ohne dass einer dieser Punkte ihn sieht?

### Vorbereitete Daten (play-Baseline, 480 s, BVH an, alle play-Schalter)
Läufe: PC 1 `20261126-192359` (3070), PC 2 `20261126-193216-play-base` (4070). JOURNAL "senior 43".

| Größe pro Frame (Median Welt) | PC 1 | PC 2 | Marke / Quelle |
|---|---|---|---|
| Frame = Vorb. + GPU + Zurück | 344,1 = 112,6 + 182,6 + 46,5 ms | 447,6 = 144,3 + 158,8 + 64,3 ms | [M] host-Zeile |
| Submissions | 21,1 | 19,0 | [M] copystat |
| Refresh-Anforderungen | 22.939, 16.511 MB angefragt | 23.010, 16.415 MB | [M] copystat (`Step::Refresh`, host.rs:2492) |
| davon wirklich kopiert | 146,6 MB in 38,3 ms | 125,7 MB in 51,4 ms | [M] host-Zeile "copies" |
| angefragt / kopiert | 112,7x | 130,6x | [C] |
| Änderungsprüfungen ("watches") | 4.930, 5.402 MB geprüft, 97,0 % unverändert, 11,7 ms | 4.377, 5.626 MB, 97,8 %, 12,8 ms | [M]/[C] (`update_tracked`, shadow.rs:1118) |
| GetWriteWatch-Kernelzeit | 0,0 ms | – | [M] copystat "watch time: kernel" |
| Guard-Faults (Titel-CPU-Writes auf bewachte Seiten) | 1.148 | 898 | [M] (`on_write`, watch.rs:1131/1150) |
| Handler-Zeit gesamt / je Fault | 73,4 ms / 64,0 µs | 56,9 ms / 63,3 µs | [M]/[C] (inkl. Warten auf Guard-Lock) |
| Seitenfehler gesamt | 2.088 | 10.850 | [M] |
| Schutzwechsel: Aufrufe / Seiten geschützt / geöffnet | 2.606 / 3.142 / 2.278 | 2.359 / 2.716 / 2.081 | [M] |
| Titel-beschriebene Seiten (distinct) / davon schon im Vorframe | 1.505 / 70,0 % | 1.259 / 65,8 % | [M]/[C] (`frame_done`, watch.rs:774) |
| Guard-Lock gehalten: Handler / take_split / taken / writing | 9,8 / 8,7 / 4,3 / 3,6 ms | – | [M] pc1-eval (watch.rs:1147/510/553/812) |
| GPU-markierte Seiten verglichen / zurückgeschrieben | 2.900 / 104,1 MB in 38,3 ms | 2.865 / 102,1 MB in 55,5 ms | [M] (`copy_back_marks`, shadow.rs:1530) |
| davon über eigene View / ohne View (private Pools) | 1.000 / 667 (40,0 % ohne) | – | [M]/[C] |
| deklarierte GPU-Write-Bereiche verglichen | 63,4 MB | 53,5 MB | [M] (`gpu_writes`, shadow.rs:806) |
| Snapshots | 54,8 MB | 44,8 MB | [M] |
| Kartenkopien: hoch / geholt / Warten | 53,1 / 157,2 MB / 8,1 ms | 53,8 / 155,1 MB / 14,5 ms | [M] host::card |
| Heap-Slot-Sync / Titel-Threads warten auf View-Writes | 26,0 / 118,9 ms | 37,4 / 165,3 ms | [M] |

| Bestand (Median Welt) | PC 1 | PC 2 | Marke / Quelle |
|---|---|---|---|
| Host-Kopien des Titelspeichers (System-RAM) | 5.352 Kopien, 7.520 MB | 4.967, 6.118 MB | [M] bindless/"VRAM by kind" |
| davon Hulls (von Heap-Views erreichte Spanne) | 3.443 MB (45,8 %) | 3.340 MB (54,6 %) | [M]/[C] (`Viewed::ranges`, bindless.rs:147) |
| davon unordered (UAV-Spanne) | 1.008 MB | 1.007 MB | [M] |
| Kartenkopien (VRAM) | 940 MB (12,5 % der Host-Kopien) | 940 MB | [M]/[C]; Deckel 896 MiB |
| Karte belegt / Budget | 8.055 / 7.239 MB | 10.269 / 11.324 MB | [M] |

Aus anderen Läufen (nicht play):
- Arena-Modus (Branch test/bvh31-blimit, PC 2, Lauf 152953): 42.712 Titel-Writes à 4 KiB pro Frame hochgeladen,
  147 ms [M] = 166,8 MB/Frame, 3,44 µs je Stück [C]. REFRESH=0 spart 120-190 ms Vorbereitung; mit Sperre
  214 ms wegen GetWriteWatch-Kosten [M]. JOURNAL senior 31 (16:20, 16:55). Dort GetWriteWatch mit Reset
  je 4 KiB; 54 % der Arena-Seiten stammen aus NICHT write-watched Titelspeicher [M, Lauf 185839, senior 41].
- Agent B (s20, JOURNAL ~09:15): Guard-Lock seriell 26,7 ms im Copy-back, Titel-Threads warten ~50 ms im
  Handler; ~42 % markierter Seiten ohne geändertes Byte; "wird vor dem nächsten Überschreiben gelesen?"
  mit vorhandenen Zählern NICHT beantwortbar (bräuchte Lese-Guards) [M/C].
- senior 30 (JOURNAL ~09:50): Titelspeicher-Arten: `XMemVirtualAlloc` = privates VirtualAlloc2 +
  MEM_WRITE_WATCH (keine Zweitansicht); physische Seiten = Pagefile-Section, per MapViewOfFile3 gemappt
  (Aliase möglich) [C].
- senior 37 (JOURNAL 14:15/14:45): 7,5 GB Host-Kopien sind gewollt im System-RAM (CUSTOM L0 write-back);
  nächster Hebel "Kopien auf benutzte Bereiche" = Hulls 3,45 GB [M, Lauf 113916].
- Fable-Plan perf-plan.md 6.1/6.2: Prozess privat 27-28 GB, Ziel <= 20 GB; geteilter GPU-Speicher 9,9 GB,
  Ziel <= 1 GB [M].

### Kosten der Optionen (berechnet)
- Merkmal-Array: 2^24 Seiten à 64 KiB unter 2^40; 4 Bit je Seite = 8,0 MiB; + 16-Bit-4-KiB-Untermaske =
  32,0 MiB; 4 Bit je 4-KiB-Seite = 128,0 MiB [C]. Setzen per `fetch_or` ohne Lock (Vorschlag).
- Lese-Guards auf allen GPU-markierten Seiten eines Frames, je ein Fault: 2.900 x 64,0 µs = 185,5 ms (PC 1),
  181,4 ms (PC 2) [C, Eingaben M] -> nur stichprobenweise denkbar.
- Tagging im Guard-Handler unter dem Lock würde den seriellen Abschnitt (watch.rs:1147, 9,8 ms) verlängern
  -> außerhalb des Locks setzen (unverified, for Fable).

### Messpunkte im Code (play 79244da)
| Merkmal | Stelle | heute |
|---|---|---|
| CPU-Write (Titel) | watch.rs:1131 `on_write`, Lock :1147, Zähler :1150 | 64-KiB-Seite; Fault-Adresse exakt; nur erster Write je Bewachung |
| CPU-Write (Runtime/Kernel) | watch.rs:809 `writing`, :850 `retake`, :820 `title_memory_remapped` | eigene Copy-backs, Dateilesen, DirectStorage |
| Änderungen beim Refresh | watch.rs:485 `take_split`, shadow.rs:1118 `update_tracked`, :1170 Zähler | je Host-Kopie und Seite |
| Frame-Grenze | watch.rs:774 `frame_done` (written / written_before) | nur diagnostics |
| GPU-Write | shadow.rs:1530 `copy_back_marks` (Bitmap, shader.rs:88 `VIEW_WRITE_BITS` = 2^22 Seiten), shadow.rs:806 `gpu_writes`, shadow.rs:175 `card_dirty` | 64 KiB |
| GPU-Read (Proxy) | host.rs:2492 `Step::Refresh`, memory.rs:567 `prepare`, bindless.rs:147 `ranges`; echte Lesemarken nur für Textur-Heap-Slots (shader.rs:89-92) | je Kopie/Bereich, nicht je Seite |
| CPU-Read nach GPU-Write | – | nicht vorhanden |
| Regionen | victory-hle modules/xmem.rs:87 `XMemVirtualAlloc` (:136 MEM_WRITE_WATCH), :477 `XMemAllocatePhysicalPages`, :808 `map` (MapViewOfFile3, Aliase :850); shadow.rs:685 `Shadow::new_at` | – |
| Einheit | watch.rs:41-44: 64 KiB, "finer units made the checks of the world's big buffers cost more than the copies they saved" | – |
| Ungeschützt | watch.rs:14-17: C-Heap u. Ä. wird ganz kopiert (heute 0,0 MB/Frame [M]) | -> unknown |

---

## F2a (Review, nicht blockierend): Annahmen der Invarianten-Tests Q6 (`shadow_invariants.rs`)

Status: offen. Blockiert: nichts (Tests ändern keinen Produktivcode). Bitte nur die Annahmen prüfen:
1. GPU-Writes werden simuliert (`write_host` in die Host-Kopie), ohne echte Submission; nur
   `a_title_write_reaches_the_gpu_side_of_a_copy_in_gpu_memory` läuft wirklich auf der GPU (Kartenkopie).
2. Markierte Seiten gehen über `copy_back_marked` je Seite zurück (wie `copy_back_marks` es aufruft), nicht über
   die globale Bitmap (sonst Wettlauf mit parallelen Tests).
3. Für markierte Writes wird wie in draw.rs/bindless.rs der auf Seiten erweiterte Bereich aufgefrischt
   (`page_range`). Ohne das fielen 10 von 64 Abläufen durch: Teilseiten bleiben "changed", `copy_back_marked`
   verwirft dann die GPU-Bytes (dokumentierter shortcut). **unverified, for Fable:** gibt es Aufrufer, die markierte
   Writes deklarieren, ohne den Seitenbereich aufzufrischen?
4. Seiten, die Shader an berechneten Adressen schreiben (`copy_back_page`), sind nicht armiert: schreibt der Titel
   dieselbe Seite während die GPU läuft, gehen die GPU-Bytes verloren (dokumentierter shortcut, einmal geloggt).
   Test (b) schreibt für diesen Weg deshalb eine andere Seite. **unverified, for Fable:** für Stufe 3 relevant?
5. Gerüste (d) für Z2: Erwartungen stehen in den Doc-Kommentaren; bitte bei der Umsetzung anpassen.

