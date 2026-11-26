# Performance-Plan: RTX 3070 (8 GB VRAM), 32 GB RAM

Stand: 26.11.2026, Basis `test/s32-clamp-ownview` (1a6d42e). Messrechner PC 1: RTX 3070 8 GB,
Ryzen 7 7700, 32,5 GB RAM. Fenster 1080p, BVH/RT an, Läufe 480 s.

Jede Zahl ist markiert: **[M]** von uns gemessen, **[F]** Fremdmessung (Fork-Autor, README,
upstream-PR, nicht von uns geprüft), **[S]** geschätzt, **[C]** aus Code abgeleitet.

## 0. Kurzfassung

1. Der Frame ist heute vollständig seriell: 352 ms = 134 Vorbereiten + 168 GPU + 49 Zurückkopieren [M].
2. Wichtigster neuer Befund: Der Rechner ist im Lauf **außerhalb beider Speichergrenzen** [M].
   Verfügbarer RAM im Median 0,6 GB (Minimum 0,15 GB), Prozess 27–28 GB privat, System-Commit
   56–57 GB bei 32,5 GB RAM. Der Prozess hält 6,8 GB dedizierten und **9,9 GB geteilten
   GPU-Speicher**: Mehr als die Hälfte seines GPU-Speichers liegt im Systemspeicher, nicht auf der Karte.
   Solange das so ist, ist jede Frame-Zahl mit Auslagern vermischt, und Flights (die RAM kosten) passen nicht.
3. Stufen: **5 fps erreichbar** (2–3 Wochen, [S]). **15 fps auf der 3070 nicht in Sicht**: Selbst der
   vollständige Fork-Stand ergäbe umgerechnet 10–13 fps bei RT aus und braucht 40–46 GB RAM [F/S].
   **30 fps außer Reichweite**; auch der Fork erreicht sie auf einer 7900 XTX mit 64 GB nicht (13–20 fps [F]).
4. Top 3: (1) gemeldete Auflösung senken (`VICTORY_DISPLAY_MODE`), (2) Speicherdiät für RAM und VRAM,
   (3) Zurückkopieren auf einen eigenen Lande-Thread.

Nicht gemessen, weil die Läufe vom Werkzeug abgebrochen wurden (siehe 1.4): RT aus, GPU-Zeit je Programm
auf s32, 1440p-Vergleichslauf, Streuung über mehrere s32-Läufe mit Speicherprotokoll.

## 1. Gemessene Aufschlüsselung

### 1.1 Wie gemessen wird

- Die drei Phasen sind CPU-Wanduhr auf dem einreichenden Thread, keine GPU-Zeitstempel [C]:
  Vorbereiten = bis `ExecuteCommandLists`; „GPU“ = Einreichen + Warten auf die Fence (enthält Treiber und
  Aufwachen); Zurück = Abholen der Kartenkopien + Vergleichen + Schreiben in den Titelspeicher.
- Die Summe der Phasen entspricht dem Frame (351 von 352 ms [M]): Es überlappt nichts.
  Grund im Code: Ein Lock hält eine Einreichung von der Vorbereitung bis zum Zurückkopieren
  (`host.rs`, „One submission at a time from preparation to copy-back“) [C].
- Auswertung: Median der Welt-Intervalle (>= 300 Draws/Frame), dazu p95 und Median der letzten 40.

### 1.2 RT/BVH an, 1080p-Fenster

| Build | Frame | Vorb. | GPU | Zurück | fps | Quelle |
|---|---|---|---|---|---|---|
| s20 (alte Basis), 4 Läufe | 405 (Streuung 10,7) | 135 | 198 | 70 | 2,5 | [M] Journal 11:16–11:45 |
| s29 Clamp-Fenster | 379 | 135 | 170 | 72 | 2,6 | [M] |
| s30 eigene Ansicht | 382 | 133 | 200 | 47 | 2,6 | [M] |
| s32 = s29 + s30 | 354 | 134,5 | 170,4 | 48,0 | 2,8 | [M] |
| s32, Lauf `20261126-113916` (dieser Plan) | 352,5 (p95 1134; letzte 40: 349,7) | 134,1 | 167,7 | 48,6 | 2,8 | [M] |
| r25 (upstream main + Stack) | 399 | 109 | 216 | 69 | 2,5 | [M] |

Zusätze aus dem s32-Lauf [M]: 4762 Draws/Frame, 21,3 Einreichungen/Frame, 2935 Clamp-Pässe
(70 Fenster mit 2208 Draws, 1906 Draws einzeln), 985 Seiten über die eigene Ansicht geschrieben und
647 weiter über den Guard-Weg, Guard-Lock noch ca. 26 ms/Frame gehalten (größte Stellen 8,9 + 8,1 + 3,9 + 3,2),
8635 Seitenfehler und 1109 Guard-Faults je Frame, 156,6 MB/Frame von der Karte geholt (8,4 ms Warten).

GPU-Auslastung laut `nvidia-smi` im Welt-Teil: Median 57 % [M] (5-s-Raster). Das passt zum Anteil der
GPU-Phase am Frame (168/352 = 48 %) und bestätigt erstmals mit einer Messung, dass die Karte rund die Hälfte
der Zeit wartet. Die frühere Zahl „51 % idle“ war nur aus den Phasen abgeleitet.

### 1.3 GPU-Zeit nach Befehlsart (s20, `VICTORY_GPU_TIMING=1`, Journal „finisher 14“)

Der Timing-Modus reicht jede Liste einzeln ein und bläht den Frame von 400 auf 504 ms auf [M]; die
Befehlssumme (196 ms) deckt sich aber mit der GPU-Phase ohne Timing (196 ms) [M].

| Anteil | ms/Frame [M] | Befehle |
|---|---|---|
| ExecuteIndirect Mesh-Shader | 69,4 | 3783 |
| Compute-Dispatch | 52,6 | 2722 |
| Clamp vor ExecuteIndirect | 29,2 (in s32 bereits verkleinert: GPU −28 ms) | 5112 → 2935 |
| ExecuteIndirect VS | 14,0 | 363 |
| ExecuteIndirect Compute | 13,9 | 966 |
| Mesh direkt | 10,3 | 378 |
| Rest | 5,5 | 1100 |

Teuerste Programme [M]: Shadowmask-Checkerboard 8,2 · PedTransformPreSkin 7,9 · ScreenProbes-RayBinning 6,6 ·
GpuParticleSort 5,0 · CascadeShadows 4,9 · TransformD 4,2 · CascadeShadows-Ped 3,7 ·
light_probes RegisterAllocationRequests 3,6 (Wave64). Danach ist die Liste flach (1512 Schlüssel, keiner über 3,1 ms).
Es gibt keinen einzelnen teuren Shader; die Zeit steckt in der Zahl der Befehle.

- Wave64-Programme (165 Namen) zusammen 18,9 ms/Frame [M]; der Anteil der Aufteilung ist nicht trennbar.
  Ohne Aufteilung höchstens ca. −9 ms [S], und das nur auf einem 64-Lane-Host.
- Atomics: Die Schreibmarken unserer Shader (#1962-Schema) kosten nach dem Fix „Laden vor Atomic“ ca. 1 ns
  je Marke [M, synthetisch]; ein ms-Anteil je Frame ist nicht gemessen. Die Bindless-Variante von #1962
  hatte GPU ca. 260 ms gegen 144–190 ohne [M, älterer Stand], Ursache unbelegt.
- RT an gegen aus: eingeschwungen BVH-Builder 1,0 + RT-Leser 0,9 ms/Frame [M, GPU-Zeitstempel]. Der Fork-Autor
  nennt für seinen Builder dagegen ca. +40 ms GPU [F]. Ein direkter A/B-Lauf RT aus auf s32 fehlt (1.4).
  Erwartung nach unserer Messung: RT aus spart eingeschwungen unter 5 ms, in Ladephasen viel
  (Builder 500–950 ms/Frame [M]).

### 1.4 Was nicht gemessen werden konnte

Geplant waren sechs Läufe (2x RT an, RT aus 1080p, RT aus 1440p, 2x GPU-Timing). Fertig wurde einer. Befehle
über 180 s werden vom Werkzeug in den Hintergrund verschoben, und Hintergrund-Shells werden bei knappem Speicher
beendet. Knapp ist der Speicher wegen des Spiels selbst (Abschnitt 6). Abhilfe: Claude Code mit
`CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1` starten (nur der Nutzer kann das), oder zuerst die
Speicherdiät. Offen bleiben damit: RT aus (1080p), GPU-Zeit je Programm auf s32, Streuung des
Speicherverlaufs.

Zu „1440p“: Einen Schalter gibt es in unserem Baum nicht. Wir melden dem Titel fest 3840x2160
(`dxgi.rs`, `modes`) [C]; der Titel rendert daraufhin Ziele in 2560x1440 [M, Capture-Größen im Journal].
Die „1080p“ sind nur die Fenstergröße. Der Fork meldet standardmäßig dasselbe; seine „1440p“ sind sein
Monitor [C]. Die interne Auflösung ist also schon heute vergleichbar mit seiner.

## 2. Fairer Vergleich mit dem AMD-Fork

| | Fork (avatarmanz) | Wir (s32) |
|---|---|---|
| Hardware | RX 7900 XTX 24 GB, i9-12900K, 64 GB | RTX 3070 8 GB, Ryzen 7 7700, 32 GB |
| Szene | Prolog, innen | Welt-Intervalle des Standardlaufs (Szene nicht gleich) |
| RT | aus (Ersatz `VICTORY_AMBIENT=0.8`) | an |
| Frame | 46–65 ms, 13–20 fps [F] | 352 ms, 2,8 fps [M] |
| Karte beschäftigt | ca. 41 ms [F] | 168 ms Fence-Phase [M] |
| Vorbereiten | 48 ms auf eigenem Thread, überlappt [F] | 134 ms seriell [M] |
| Zurück | auf Lande-Threads, überlappt [C] | 49 ms seriell [M] |
| Prozess-RAM | 40–46 GB privat [F] | 27–28 GB privat [M] |

Einordnung:

- Hardware: Eine 7900 XTX ist bei Rasterlast in 1440p grob 1,8–2,0x so schnell wie eine 3070 [S, allgemeine
  Benchmarks, nicht an diesem Titel gemessen]. Seine 41 ms wären auf der 3070 also 75–85 ms [S].
- GPU-Abstand: 168 gegen 41 ms = 127 ms. Davon ca. 40 ms Hardware [S], der Rest (ca. 85 ms) Software und Speicher:
  2935 Clamp-Pässe gegen seine 133 [M/F], Shader ohne Aufräum-Pässe (in der fxceless-Basis −43 % Instruktionen,
  ca. −20 ms auf einer 4070 [F]), unsere Schreibmarken in den Shadern, und 9,9 GB GPU-Speicher außerhalb der Karte [M].
- CPU-/Überlappungs-Abstand: Bei ihm liegen Vorbereiten und Zurück nicht auf dem kritischen Pfad. Bei uns
  kosten sie 134 + 49 = 183 ms des Frames [M].
- Fazit: Von ca. 297 ms Abstand (352 gegen ca. 55) entfallen rund 60 % auf fehlende Überlappung/CPU und rund
  40 % auf GPU-Zeit, davon etwa ein Drittel reine Hardware [S aus M- und F-Zahlen].
- Näher an unserer Hardware [F]: upstream #2712 (RTX 3070, fxceless) 323 ms; fxceless-README (RTX 4070)
  420–450 ms Standard, 240–350 ms mit Relay; #3081 (4070 SUPER) 2 fps → 5 fps bei 720p. Auf NVIDIA mit 8–12 GB
  hat bisher niemand mehr als ca. 5 fps gezeigt. Die „16 fps auf NVIDIA“ waren ein Lesefehler (16,7 Sekunden je Frame).

Der direkte Lauf „RT aus, innen“ auf PC 1 fehlt (1.4). Er ändert die Einordnung voraussichtlich wenig, weil RT
eingeschwungen unter 5 ms kostet [M].

## 3. Techniken aus dem Fork

Lizenz: Der Fork ist GPL-3.0, einzelne Dateien (`console_insn*.rs`) GPL-2.0-only und tabu. Die Tabelle beschreibt
Verfahren; ob Code übernommen oder neu geschrieben wird, ist eine Entscheidung des Nutzers.
Gewinne sind [F] (auf seiner Hardware) oder [S]; „Warm-up“ = erzeugte Shader ändern sich.

| # | Technik (Schalter) | Was sie tut | Haben wir? | Übertragbar auf 3070 | Aufwand | Gewinn | RAM/VRAM |
|---|---|---|---|---|---|---|---|
| T1 | Flights (`LATE_WAIT` 0–3, `LATE_WAIT_LISTS` 256) | Einreichen ohne Warten, Abrechnen erst an Beobachtungspunkten | nein | ja, herstellerneutral | L | Fence-Warten und Überlappung; Kern von 124 → 75 ms [F] | je Liste Scratch >= 256 KiB, Upload- und Deskriptor-Slots bis zur Landung [C] |
| T2 | Relay-Thread (`RELAY=2`, `RELAY_COPY`) | Landen fertiger Flights auf eigenem Thread, Leser warten nur auf betroffene Flights | nein | ja | L (braucht T1) | „Leerlauf der Karte 136 → 30 ms“ [F, 4070] | ein Readback-Puffer in Größe des größten Abholens |
| T3 | Paralleles Landen neben den Guards (`LAND*`, `SAFE_COPY`) | Schreiben über eigene RW-Ansicht, kein Schutzwechsel, kein Guard-Lock, Notiz danach gebündelt | teilweise (`BACK_OWN_VIEW`, `BACK_THREADS`) | ja | M | Zurück 49 → überlappt | nur Adressraum |
| T4 | Breite Snapshots/Vergleiche, AVX2 (`WIDE_BACK`, `UNDER_WIDE`) | 1-MiB-Jobs auf festen Threads, nur geänderte dwords | teilweise | ja | M | 14 → 4–7 ms [F] | Snapshot-Pool bis 512 MiB |
| T5 | Kartenkopien mit Hash auf der Karte (`CARD_REHASH_RUNS`) | Shader-Schreibzugriffe per Hash-Pass finden statt per Marke im Shader | teilweise (Kopien ja, Hash nein) | ja, Grenzen skalieren mit VRAM (3070: 512 MiB) | L, Warm-up | Karte 79 → 66 ms [F] | VRAM 2x Kopie bei beschriebenen Puffern |
| T6 | THIN (`THIN`, 8 Threads) | Listenaufrufe nur protokollieren, parallel übersetzen | nein | ja, aber RAM-teuer | L | 75 → 68 ms [F] | „lists' allocators 9 GB“ [F] |
| T7 | Asynchrones Einreichen (`SUBMIT_DEPTH` 64, `SUBMIT_FRAMES` 1) | Titel-Aufruf kehrt sofort zurück, eigener Thread bereitet vor | nein | ja | L (braucht T1/T2) | 103 → 75 ms [F] | Sendebytes je Einreichung (76,6 MB/Frame [C]) |
| T8 | `JOIN_PARTS=8` | mehrere Einreichungsteile als einer | nein | ja | M nach T7 | 54,8 → 51,4 ms [F] | – |
| T9 | `CARD_STEPS` | Füllen/Kopieren auf der Karte statt auf Landung warten | nein | ja | L | 25–30 von 105 ms [F] | – |
| T10 | `UP_QUEUE`, `COPIES_IN_PLACE`, `SEND_OWN` | Uploads über Kopier-Queue, Kopien auf der Karte | nein | ja | M | ca. 5 ms [F] | ein Staging-Puffer |
| T11 | Clamp-Fenster und -Batches (`INDIRECT_WINDOW`, `INDIRECT_BATCH=2`) | Clamps gesammelt, nur für später benötigte Records | Fenster ja (`CLAMP_WINDOW`), Batches nein | ja; `INDIRECT_WIDE` nur AMD (Korrektheit) | M, kein Warm-up [C] | 84,6 → 75,3 ms [F]; bei uns noch 2935 Pässe | klein |
| T12 | Heap-Sync nebenher, lazy Tabellen-Records (`HEAP_SYNC_ASIDE/EARLY`) | Sync auf eigenem Thread; Records, die nur in Draw-Tabellen landen, macht der Draw | nein | ja | M | 280 → 124 ms [F]; bei uns Sync 25 ms/Frame [M] | – |
| T13 | `PORT_DESCRIPTORS` | Titelcode patchen, Slots melden statt Seiten scannen | nein | ja, feste Image-Offsets | L | unbekannt | – |
| T14 | Guard-Kosten (`PAGE_PARTS`, `GUARD_REST`, `OPEN_ONE_VIEW`) | weniger Faults und Schutzwechsel | nein | ja | M | Titel-Threads 69 → 6 ms [F]; bei uns 1109 Guard-Faults/Frame [M] | – |
| T15 | Texturpfad (`UPLOAD_AHEAD`, `TEXTURE_POOL`, `LEVELS_ON_DEMAND`, `TEXTURE_KEEP`) | Erst-Uploads über Kopier-Queue, 64-MiB-Blöcke, Mips nach Bedarf | nein | teils: Pool/Keep sind für 24 GB ausgelegt; `LEVELS_ON_DEMAND` spart VRAM | M je Teil | Ruckler, VRAM | Pool hält bis 4 GiB leer [C]: auf 8 GB abschalten |
| T16 | Threads/Takt (`CARD_AWAKE`, `FAST_CORES`, Prioritäten) | Mini-Last hält GPU-Takt, Threads pinnen | nein | `CARD_AWAKE` ja; `FAST_CORES` für Hybrid-CPUs, auf dem 7700 ohne Wirkung | S | „30 ms am Stück statt 38–40“ [F] | – |
| T17 | Pipeline-Buch, `victory warm` | Ruckler beim Erzeugen | nein | ja | M | kein Dauergewinn | – |
| T18 | `SHADER_OPT` (fxceless-Basis) | Aufräum-Pässe über erzeugte Shader | nein | ja | M, Warm-up | −43 % Instruktionen, ca. −20 ms GPU auf 4070 [F] | – |
| T18b | `RDNA_FLOAT`, natives Wave64 | nur auf RDNA aktiv | nein | **nein (AMD-Hardware)** | – | 0,8 ms [F] | – |

Was an AMD-Hardware hängt und was die NVIDIA-Seite stattdessen hat:

- **Natives Wave64**: RDNA führt Konsolen-Wave64-Compute direkt aus. Auf 32 Lanes hat auch der Fork nur
  Näherungen; seine exakte Zwei-Wellen-Form (`WAVE_WIDE`) ist aus, weil sie „die Karte verlor“ [F].
  Unsere Alternative: die eigene Split-Emulation (vorhanden) und upstream #3147 (zwei Host-Wellen, Austausch über
  group-shared memory; laut PR nur auf einer RTX 5060 Laptop gelaufen, Bild dort noch schwarz, kein Ampere-Lauf [F]).
  Möglicher Gewinn höchstens ca. 9 von 18,9 ms [S]; kein früher Kandidat.
- **`INDIRECT_WIDE`**: Dispatch-Zähler über 65.535 nur auf AMD. Auf NVIDIA bleibt der Schnitt; das ist Bildkorrektheit
  („Blocky patches“), nicht Tempo.
- **24 GB VRAM**: `TEXTURE_POOL`, `TEXTURE_KEEP`, `KEEP_FREED`, `CARD_ROOM_FOR_BUFFERS` halten Speicher bis 85–88 %
  des Budgets. Auf 8 GB sind das Risiken, keine Gewinne.
- **64 GB RAM**: T6/T7 erklären den Großteil seiner 40–46 GB („allocators 9 GB, windows 7, uploads 4“ [F]).

## 4. Fix-Liste nach Rang

| Rang | Ursache | Gemessen | Fix | Erwarteter Gewinn | Aufwand | Warm-up | Bildrisiko |
|---|---|---|---|---|---|---|---|
| 1 | Titel rendert 2560x1440, weil wir 3840x2160 melden; 9,9 GB GPU-Speicher liegen außerhalb der Karte | geteilter GPU-Speicher 9,9 GB, GPU-Phase 168 ms [M] | `VICTORY_DISPLAY_MODE=WxH` in `dxgi.rs` (1920x1080, 1280x720) | GPU −30…−70 ms, VRAM −1…−3 GB [S]; #3081: 2 → 5 fps bei 720p [F] | S (0,5 Tag) | nein [C] | mittel: neue Auflösung, Capture-Vergleich braucht neue Referenz; unbelegt, ob der Titel der Meldung folgt |
| 2 | Prozess 27–28 GB privat, verfügbar 0,6 GB, System-Commit 57 GB | [M] | Speicherdiät (Abschnitt 6): Pipeline-Library per mmap, eine statt drei Kopien je Placed-Textur, Texturen, die nur über Heap-Slots sichtbar sind, verzögert/kleiner anlegen | −6…−10 GB RAM, kein Auslagern; Frame-Gewinn unbekannt, p95 sinkt [S] | M (3–5 Tage) | nein | niedrig bis mittel (Texturen zu spät sichtbar) |
| 3 | Zurückkopieren seriell auf dem einreichenden Thread | 48,6 ms [M]; Guard-Lock ca. 26 ms gehalten [M] | Lande-Thread (T3-Verfahren): nächste Vorbereitung startet sofort, Landen läuft daneben über die eigene Ansicht; Leser warten nur auf betroffene Bereiche | −35…−45 ms → ca. 310 ms [S] | M–L (4–6 Tage) | nein | mittel: Titel liest Speicher, der noch nicht gelandet ist |
| 4 | Heap-Sync inline | 24,9 ms/Frame, 446.000 Slots [M] | Sync auf eigenem Thread + lazy Tabellen-Records (T12) | −10…−20 ms [S]; die reine Überlappung brachte −4 (nicht signifikant) [M] | M (2–3 Tage) | nein | mittel (veralteter Slot) |
| 5 | Noch 2935 Clamp-Pässe, 1906 Draws „may store“ einzeln | [M] | Clamp-Batches (T11): nur die später gebrauchten Records, Tabelle beim Aufzeichnen | −8…−15 ms GPU [S] | M (2–3 Tage) | nein [C] | mittel (Clamp fehlt → Watchdog, wie bei UNCLAMPED gemessen) |
| 6 | Vorbereiten seriell vor der GPU | 134 ms [M] | Flights Tiefe 1 (T1/T2, ohne THIN): Vorbereiten von N+1 während GPU von N | Frame → max(GPU, Vorb.) + Rest, ca. 170–200 ms [S] | L (2 Wochen) | nein | hoch (Reihenfolge, Lesen fliegender Daten) |
| 7 | Shader ohne Aufräum-Pässe | – | `SHADER_OPT`-Verfahren | −10…−20 ms GPU [F/S] | M, **voller Warm-up** | ja | mittel (alle Shader neu) |
| 8 | 1109 Guard-Faults, 8635 Seitenfehler je Frame | [M] | `GUARD_REST`/`PAGE_PARTS` (T14) | −5…−10 ms, weniger Streuung [S] | M | nein | niedrig |
| 9 | GPU-Takt fällt in den Pausen | Auslastung 57 % [M]; Takt selbst nicht gemessen | Takt mitloggen (`clocks.gr`), Energieprofil, erst dann `CARD_AWAKE` | 0…−20 ms [S] | S | nein | keins |
| 10 | Wave64-Aufteilung | 18,9 ms für alle Wave64-Programme [M] | #3147 abwarten, auf Ampere prüfen | höchstens −9 ms [S] | L, Warm-up | ja | hoch |

Verworfen nach Messung [M]: #2960, Heap-Sync-Dirty, Deskriptor-Cache, schlanker Clamp (LEAN=2), #2501,
8 statt 4 Copy-back-Threads, Clamp ganz weglassen (Watchdog nach 136 s). #2703 nicht in die Basis (Watchdog/TDR).

## 5. Fahrplan nach Stufen

**Die 6-fps-Grenze.** senior 27 schätzte 150–170 ms als Boden. Die Messung stützt das: Überlappen alle
CPU-Phasen vollständig, bleibt die GPU-Phase, heute 168 ms = 5,95 fps [M]. Der Clamp-Gewinn ist darin schon
enthalten. Mehr als 6 fps gibt es nur, wenn die GPU-Zeit selbst sinkt.

**Stufe 5 fps (200 ms): erreichbar.** Reihenfolge:
1. Rang 1 (Auflösung) und Rang 9 (Takt messen): je ein halber Tag, keine Architektur.
2. Rang 2 (Speicherdiät): Voraussetzung für alles, was RAM kostet, und für belastbare Messungen.
3. Rang 3 (Lande-Thread) → ca. 310 ms. Rang 4 und 5 → ca. 280 ms [S].
4. Rang 6 (Flights Tiefe 1) → 170–200 ms [S].
Wenn Rang 1 trägt (GPU unter 120 ms), sind 5 fps schon mit Rang 3–5 knapp möglich und Flights bringen 6–8 fps [S].
Aufwand gesamt 2–3 Wochen. Unsicher ist vor allem Rang 6.

**Stufe 15 fps (67 ms): auf der 3070 mit 32 GB nicht in Sicht.** Nötig wären GPU <= 60 ms und ein kritischer
CPU-Pfad <= 60 ms. Der komplette Fork-Stand ergäbe umgerechnet 75–85 ms GPU, also 10–13 fps, mit RT aus, und er
braucht 40–46 GB RAM [F/S]. Dazu kämen 720p intern, `SHADER_OPT` und ein RAM-sparsamer Ersatz für THIN.
Das sind Monate, und RT-Schatten/indirektes Licht in voller Form passen nicht dazu. Ehrliches Ziel für die
3070: 6–8 fps mit RT, 8–12 fps mit RT aus und reduzierter Auflösung [S].

**Stufe 30 fps (33 ms): außer Reichweite.** Der Fork erreicht sie auf einer rund doppelt so schnellen Karte mit
dreifachem VRAM und doppeltem RAM nicht (13–20 fps, RT aus [F]). Mit Emulation von Speicher-Guards,
Kartenkopien und Wave64 auf 32 Lanes ist das auf dieser Hardware kein realistisches Ziel.

## 6. Speicherplan

### 6.1 Gemessen (s32, Lauf `20261126-113916`, Protokoll `p-rton-1-mem.csv`, 5-s-Raster)

| Größe | 60 s | 240 s | 470 s | Median Welt | Spitze |
|---|---|---|---|---|---|
| Prozess privat (Commit) | 16,9 GB | 27,0 GB | 28,2 GB | 27,2 GB | 28,2 GB |
| Working Set | 21,1 GB | 17,9 GB | 13,1 GB | 17,8 GB | 21,4 GB |
| RAM verfügbar (System) | 0,3 GB | 0,5 GB | 0,5 GB | 0,58 GB | Minimum 0,15 GB |
| System-Commit | 46,2 GB | 55,9 GB | 57,2 GB | 56,1 GB | 57,2 GB |
| GPU dediziert (Prozess) | 6,1 GB | 6,8 GB | 6,8 GB | 6,8 GB | 6,8 GB |
| GPU geteilt (Prozess) | 2,8 GB | 9,9 GB | 9,9 GB | 9,9 GB | 10,1 GB |
| `nvidia-smi` belegt | 7,1 GB | 7,7 GB | 7,7 GB | 7,7 GB | 7,7 GB |

Lesart [M]:
- Privater Speicher wächst noch um ca. 0,3 GB/min in den letzten 5 Minuten (27,0 → 28,2 GB). Kein Plateau.
- Das Working Set **fällt**, während der Commit steigt (21 → 13 GB): Windows lagert aus. Beide harten Grenzen
  des Nutzers sind heute verletzt.
- Vor dem Start waren nur 23,4 GB verfügbar; ca. 9 GB belegen andere Programme (darunter die Agenten-Sitzungen).
- Geteilter GPU-Speicher steigt in 2 Minuten auf 9,9 GB und bleibt dort. Die Karte ist voll (6,8 von 8 GB dediziert,
  Budget 7,06–7,3 GB), der Rest liegt im Systemspeicher. Was davon je Frame gelesen wird, läuft über den Bus.
  Im Log dieses Laufs stehen keine Verdrängungs-Zeilen („evict“, „over budget“: 0 Treffer).

### 6.2 Ziele

- RAM: Prozess privat <= 20 GB, geteilter GPU-Speicher <= 1 GB, System verfügbar >= 4 GB im ganzen Lauf.
- VRAM: dediziert <= 7,0 GB (DXGI-Budget), keine Verdrängung je Frame, geteilter Anteil nahe 0.

### 6.3 Maßnahmen

| Maßnahme | Wirkung | Beleg | Aufwand |
|---|---|---|---|
| Gemeldete Auflösung senken (Rang 1) | Render-Ziele kleiner, voraussichtlich kleinere Mips | [S] | S |
| Pipeline-Library per mmap statt privat | −2,4 GB privat [F, #3147]; bei uns 3,8 GB privat [M, älterer Stand]; Datei 9,9 GB | [F/M] | M |
| Eine statt drei Kopien je Placed-Textur | bis −10 GB Commit (10,2 GB nach 80 s [M, älterer Stand]) | [M] | M |
| Texturen, die nur über Heap-Slots sichtbar sind (4,3 GB von 5,4–6,8 GB [M]), erst bei Benutzung oder mit weniger Mips anlegen (`LEVELS_ON_DEMAND`-Verfahren) | VRAM −2…−4 GB, geteilter Anteil sinkt entsprechend | [S] | M–L |
| Kartenkopien deckeln (heute 1054 MB [M]) auf 512 MiB | VRAM −0,5 GB | [C] | S |
| Fork-Pools (`TEXTURE_POOL`, `KEEP_FREED`) nicht übernehmen | vermeidet +1…4 GB | [C] | – |
| Speicherprotokoll als fester Teil jedes Laufs | Regressionen sichtbar | – | S |

### 6.4 Flights innerhalb von 32 GB

- Heute passt **kein** zusätzlicher Frame in der Luft: verfügbar 0,6 GB [M].
- Kosten je Flight [C]: Scratch mindestens 256 KiB je Liste (256 Listen = 64 MiB), Sendebytes ca. 77 MB je Frame,
  Snapshot-Pool gedeckelt bei 512 MiB, dazu Upload-Arenen. Tiefe 1 kostet damit ca. 0,2–0,8 GB [S].
  Die 40–46 GB des Forks kommen nicht aus der Tiefe, sondern aus THIN und den Listen-Allokatoren (9 + 7 + 4 GB [F]).
- Empfehlung: nach der Diät **Tiefe 1** (ein Frame in der Luft, wie sein `SUBMIT_FRAMES=1`), höchstens 64 Listen
  in der Luft, kein THIN. Budget: 20 GB Prozess + 1 GB Flights.
- Was die Grenze kostet: nichts Messbares. Der Titel wird vom Present getaktet; ab Tiefe 1 ist der Frame
  max(GPU, Vorbereiten). Mehr Tiefe glättet nur, sie erhöht den Durchsatz nicht [S]. Der Verzicht auf THIN kostet
  den Anteil, den er dem Fork brachte (75 → 68 ms, ca. 10 % [F]).

## 7. Der beste nächste Fix und sein Testplan

**Empfehlung: zuerst `VICTORY_DISPLAY_MODE` (Rang 1), direkt danach der Lande-Thread (Rang 3).**
Der Lande-Thread war der erwartete Kandidat und bleibt der beste Architektur-Schritt (bis −45 ms). Der
Auflösungsschalter kommt davor, weil er einen halben Tag kostet, GPU-Zeit und VRAM zugleich angreift, kein
Warm-up braucht und sein Ergebnis bestimmt, wie viel Flights später wert sind.

Testplan Rang 1 (PC 1, s32 + Schalter, gleiche Umgebung, `VICTORY_CLAMP_WINDOW=1 VICTORY_BACK_OWN_VIEW=1 VICTORY_BACK_THREADS=4`):
1. A = s32 ohne Schalter, B1 = 1920x1080, B2 = 1280x720. Reihenfolge A B1 A B2 A B1 B2: 3x A, je 2x B, 480 s.
2. Je Lauf: Median, p95, letzte 40, drei Phasen, Draws/Frame, Speicherprotokoll (privat, verfügbar, GPU dediziert/geteilt).
3. Prüfen, ob der Titel folgt: Größe der Render-Ziele im Capture. Bleibt sie bei 2560x1440, ist der Schalter wirkungslos.
4. Gewertet wird nur ein Unterschied über 11 ms (A-A-Streuung 10,7 ms [M]).
5. Bild: Capture-Frames 100–800 gegen A auf gleiche Inhalte prüfen (andere Auflösung, also Sichtprüfung auf
   fehlende Pässe, schwarze Ziele, Blockfehler); kein Absturz in 8 min; `cargo test --workspace`.

Testplan Rang 3 (Lande-Thread):
1. Schalter `VICTORY_BACK_LAND=1`, Standard aus. A = s32, B = s32 + Schalter. Reihenfolge A B A B A B, je 480 s.
2. Erwartung: Frame = Vorb. + GPU + Rest, Zurück-Phase auf dem einreichenden Thread unter 10 ms; Ziel <= 315 ms.
3. Korrektheit: gleiche Capture-Frames wie A pixelgenau bis auf bekanntes Rauschen; Zähler „Leser musste auf
   Landung warten“ je Frame; Guard-Faults dürfen nicht steigen; RAM-Protokoll (+ weniger als 0,5 GB).
4. Abbruch: Bildfehler auf den festen Frames oder Gewinn unter 11 ms nach 3 Paaren → Befund notieren, nicht in die Basis.

Voraussetzung für beide: Läufe dürfen nicht mehr vom Werkzeug beendet werden (1.4).
