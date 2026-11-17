# Legende: was in den Tracking-Daten steht

Quelle: die Metadaten `BAR-ATH_tactical_atd.xml` und DATA.md. Diese Datei beschreibt nur die
Struktur, keine Spieldaten. Sie darf deshalb ins Repo.

`python -m src.inspect_data --describe` gibt dieselbe Struktur direkt aus der XML aus.

---

## 1. Welche Datei was enthält

| Datei | Inhalt |
|---|---|
| `tactical/BAR-ATH_tactical_atd.txt` | **Die Messwerte.** Eine Zeile pro Frame: Positionen und Tempo von 46 Spuren und vom Ball |
| `tactical/BAR-ATH_tactical_atd.xml` | **Die Legende zur .txt.** Teams, Spuren, Einheiten, Aufbau einer Zeile |
| `tactical/BAR-ATH_tactical_smart_tagging.xml` | Automatisch erkannte Spielphasen und Standards (noch nicht eingebaut) |
| `tactical/BAR-ATH_tactical_homography.csv` | Umrechnung Spielfeld ↔ Video-Pixel (noch nicht eingebaut) |
| `tactical/BAR-ATH_tactical.mp4` | Das Video |
| `broadcast/…` | Dasselbe für die TV-Kamera, **nicht** synchron zum taktischen Video |

---

## 2. Eine Zeile der Rohdatei (`.txt`)

```
<frame> : <x>,<y>,<s> ; <x>,<y>,<s> ; … ; <x>,<y>,<s> : <x>,<y>,<s>
          └ Spur 1 ─┘   └ Spur 2 ─┘       └ Spur 46 ┘   └─ Ball ──┘
```

Trennzeichen: `:` trennt Frame, Spieler und Ball. `;` trennt die Spuren. `,` trennt die Werte
innerhalb einer Spur.

| Feld | Bedeutung | Einheit / Bereich |
|---|---|---|
| `frame` | Frame-Nummer in der Datei | `1 … 199319`. **Video-Frame = frame − 1** |
| `x` | Position entlang des Spielfelds (Torlinie zu Torlinie) | 0 bis 1, `0` = linke Torlinie |
| `y` | Position quer zum Spielfeld | 0 bis 1, `0` = obere Seitenlinie im Bild (einmal gegen das Video prüfen) |
| `s` (`speed`) | Geschwindigkeit, nur der Betrag, keine Richtung | **km/h** laut XML (`kph`, „Speed in kilometers per hour“), siehe Hinweis unten |
| Ball `x`, `y` | Ballposition | wie oben |
| Ball `s` | Ballgeschwindigkeit | wie oben, km/h |
| leer oder `NaN` | nicht erfasst (Spur gerade nicht aktiv, Ball nicht gesehen) | – |

> **Achtung, Einheit:** DATA.md schreibt „speed in m/s“, die XML sagt km/h (`kph`). Prüfen mit
> `python -m src.data_loader`: Die Zeile `player speed` zeigt Median, 99. Perzentil und Maximum.
> Sprints liegen bei etwa **30–36 km/h** (= 8–10 m/s). Ein Maximum um 30–36 heißt km/h, eines um
> 8–10 heißt m/s. Umrechnung: km/h ÷ 3,6 = m/s.

Eine Ballhöhe (`z`) gibt es nicht. Die Daten sind rein zweidimensional.

---

## 3. Die Metadaten (`.xml`): was für dieses Spiel drinsteht

| Abschnitt | Inhalt | Für uns wichtig |
|---|---|---|
| `GlobalConfig` | Bildrate `25.0`, Anbieter Metrica Sports | 25 Frames = 1 Sekunde |
| `GlobalConfig` → Halbzeit-Parameter | `first_half_start=0`, `first_half_end=199319`, 2. Halbzeit leer | **Nicht gepflegt.** Die echten Phasen stehen in `config.yaml` (Abschnitt 7) |
| `Sessions` → `Score` | welche Team-ID Heim (`idLocalTeam`) und welche Auswärts (`idVisitingTeam`) ist | daher kommen `H` und `A` |
| `Sessions` → `FieldSize` | 105 × 68 | wir rechnen mit 105,3 × 68,0 m (wie die offizielle `games.py`) |
| `Teams` | **Barcelona** (Heim, `H`): spielt in der 1. Halbzeit **links → rechts**. **Atletic Club** (Auswärts, `A`): **rechts → links** | in der 2. Halbzeit getauscht |
| `Players` | 46 „Spieler“, die in Wahrheit **Spuren** sind: `Track 1` … `Track 46` | siehe unten |
| `Sensors` | `position` mit Kanälen `x`, `y` (normalisiert). `speed` mit Kanal `s` (kph) | die Messgrößen |
| `PlayerChannels` | `player<n>_x`, `player<n>_y`, `player<n>_speed` gehören zu Spur `n` | Zuordnung Wert → Spur |
| `DataFormatSpecifications` | ein einziges Zeilenlayout für Frames 1–199319 (Abschnitt 2) | – |

**Die Spuren (`Players`):**

| Spuren | Team | Hinweis |
|---|---|---|
| 1–23 | Barcelona (`H`) | `position_type = Unknown` |
| **24** | Barcelona (`H`) | als **Goalkeeper** markiert |
| 25–45 | Atletic Club (`A`) | `position_type = Unknown` |
| **46** | Atletic Club (`A`) | als **Goalkeeper** markiert |

- `ShirtNumber` ist nur die Spurnummer, **keine Rückennummer**.
- Eine Spur ist **keine feste Person**. Spuren brechen ab, tauchen neu auf und können die Person
  wechseln. Zu jedem Zeitpunkt sind nur etwa 22 der 46 Spuren aktiv.
- Spur 24 (H) und 46 (A) sind laut XML die Torhüter (`position_type` Goalkeeper). Die Pipeline
  nimmt sie nur, solange sie in der eigenen Hälfte stehen, siehe Abschnitt 4.
- Die Spieler-Parameter `position_x` und `position_y` sind Platzhalter (0,0 bzw. 0,5 beim
  Torwart). Nicht verwenden.

---

## 4. Im Code: das `Tracking`-Objekt (`load_tracking`)

Index `i` ist überall Video-Frame `i` = Zeile `i` der Abgabe.

| Feld | Form | Inhalt |
|---|---|---|
| `ball` | (Frames, 2) | Ball `x`, `y`, 0 bis 1 |
| `ball_speed` | (Frames,) | Ball `s`, in `speed_unit` |
| `ball_z` | (Frames,) | Ballhöhe. Im Export nicht enthalten, also immer `NaN` |
| `players` | (Frames, 46, 2) | `x`, `y` jeder Spur |
| `speed` | (Frames, 46) | `s` jeder Spur, in `speed_unit` |
| `speed_unit` | Text | Einheit laut XML: `"kph"` |
| `team` | (46,) | `"H"` oder `"A"` pro Spur |
| `slot_ids` | (46,) | Spurnummern `"1"` … `"46"` |
| `team_names` | (2,) | `["Barcelona", "Atletic Club"]` |
| `home_attacks_first_half` | Text | `"left_to_right"` |
| `first_frame` | Zahl | Video-Frame von Index 0: `0` fürs ganze Spiel, bei einem Teilstück dessen Start |
| `frames` | (Frames,) | Video-Frame jedes Index (berechnet aus `first_frame`) |

`tracking.slice(a, b)` schneidet die Video-Frames `a` bis `b` als eigenes `Tracking` heraus. Die
Arrays sind Ansichten auf die Originaldaten, es wird nichts kopiert.

**In der Pipeline sind die Spieler bereinigt** (`data.tracking`, `src/player_cleaning.py`); die
Rohdaten liegen in `data.raw_tracking`. Lücken bis 10 s werden überbrückt, auch wenn eine neue Spur
den Spieler übernimmt. In den Halbzeiten kommen noch fehlende Spieler aus Video 2 dazu, in
Zusatz-Slots `v2H1` bis `v2H5` und `v2A1` bis `v2A5`. Woher jede Position stammt, steht in
`data.player_source`: `0` keine, `1` gemessen, `2` überbrückt (gleiche Spur), `3` überbrückt
(andere Spur), `4` aus Video 2, `5` Torwart = tiefster Spieler, `6` Torwart an letzter Position gehalten.

**Torhüter** (`src/goalkeeper.py`, Einstellungen unter `goalkeeper` in config.yaml): jedes Team hat
in jedem Frame der Halbzeiten genau einen Torwart, in Slot `GKH` bzw. `GKA` (die laut XML als
Goalkeeper markierte Spur, umbenannt; fehlt sie, ein neuer Slot). Die Position, in dieser
Reihenfolge:

1. die Torwart-Spur, wenn sie in der eigenen Hälfte steht (kurze Lücken davon vorher überbrückt,
   nur mit sich selbst);
2. sonst der tiefste Spieler des Teams, wenn er im eigenen Strafraum steht und mindestens
   `margin_m` (5 m) tiefer als der nächste Mitspieler; er wandert dann in den Torwart-Slot
   (Code `5`);
3. sonst bleibt er, wo er in dieser Halbzeit zuletzt gesehen wurde, und springt auf die neue
   Position, sobald er wieder auftaucht (Code `6`). Vor seiner ersten Sichtung einer Halbzeit
   steht er dort, wo er zuerst gesehen wird; nie gesehen: `fallback_depth_m` (5 m) vor der Tormitte.

Die gehaltenen Positionen sind keine Messungen: die Indikatoren für tote Bälle und Block-
Verschiebungen lassen sie weg (`measured(data)` in `src/pipeline.py`).

Welches Tor ein Team verteidigt, kommt aus der Aufstellung beim Anstoß (alle in der eigenen
Hälfte) und wird mit `home_attacks_first_half` abgeglichen. Im Viewer sind Torhüter Quadrate mit
„GK“, gehaltene hohl.

---

## 5. Der DataFrame (`to_dataframe`): eine Zeile pro Frame

| Spalte | Inhalt | Herkunft |
|---|---|---|
| `frame` (Index) | Video-Frame = Zeile der Abgabe | Datei-Frame − 1 |
| `time_s` | Sekunden seit Videostart | berechnet: Frame ÷ 25 |
| `phase` | Spielphase, z. B. `first_half` (Abschnitt 7) | aus `config.yaml` |
| `half` | `1`, `2` oder `0` (außerhalb der Halbzeiten) | aus `config.yaml` |
| `ball_x`, `ball_y` | Ballposition, 0 bis 1 | Datei |
| `ball_s` | Ballgeschwindigkeit, km/h | Datei |
| `<n>_x`, `<n>_y` | Position von Spur `n`, 0 bis 1 | Datei |
| `<n>_s` | Geschwindigkeit von Spur `n`, km/h | Datei |

`df.attrs["team"]` ordnet jede Spur `H` oder `A` zu. `df.attrs["speed_unit"]` ist die Einheit.

## 6. Die Tabelle pro Frame (`frame_table`): eine Zeile pro Objekt

| Spalte | Inhalt |
|---|---|
| `object` | `ball` oder Spurnummer |
| `team` | `H`, `A`, beim Ball `-` |
| `x`, `y` | Position, 0 bis 1 |
| `x_m`, `y_m` | Position in Metern (× 105,3 bzw. × 68,0) |
| `speed_kph` | Geschwindigkeit aus der Datei |
| `dist_to_ball_m` | Abstand zum Ball in Metern (berechnet). `NaN`, wenn der Ball fehlt |

---

## 7. Spielphasen

Definiert in `config.yaml` unter `phases`, umgesetzt in `src/phases.py`. Video-Frames, erster und
letzter Frame jeweils inklusive. Die Phasen decken alle 199.319 Frames lückenlos ab.

| Phase (`phase`) | Frames | Anzahl | Dauer | bewertet | `half` |
|---|---|---|---|---|---|
| `pre_match` | 0 – 11.184 | 11.185 | 7:27 | nein | 0 |
| `first_half` | 11.185 – 90.651 | 79.467 | 52:58 | ja | 1 |
| `halftime` | 90.652 – 110.807 | 20.156 | 13:26 | nein | 0 |
| `second_half_pre_kickoff` | 110.808 – 113.741 | 2.934 | 1:57 | ja, Ground Truth ist durchgehend `D` | 2 |
| `second_half` | 113.742 – 190.197 | 76.456 | 50:58 | ja | 2 |
| `post_match` | 190.198 – 199.318 | 9.121 | 6:04 | nein | 0 |

Bewertet sind zusammen 158.857 Frames. Aufteilen:

```python
from src.phases import split_tracking, split_dataframe
parts = split_tracking(tracking, config)     # {"first_half": Tracking, ...}
dfs = split_dataframe(df, config)            # {"first_half": DataFrame, ...}
```

`python -m src.phases` zeigt die Phasen mit Dauer, Ballsichtbarkeit und erfassten Spuren pro Frame.

`python main.py` erledigt Laden, DataFrame und Aufteilen in einem Schritt und zeigt die Array-Formen
pro Phase. Mit `python -i main.py` bleibt man danach in Python und hat alles in `data`:
`data.parts["first_half"]` (Arrays) und `data.dfs["first_half"]` (DataFrame).

## 8. Ballanalyse (`data.ball`, `src/ball_analysis.py`)

**`data.ball` ist die Balldatenbasis.** Sie entsteht in `main.py` aus dem **zusammengeführten** Ball:
Video 1, ergänzt um Video 2 in langen Lücken (Abschnitt 10). Danach läuft die Analyse unten darüber.
Ohne TV-Daten kommt der Ball aus Video 1 allein.

`analyze_ball` bekommt den Ball-DataFrame (`ball_dataframe(df)`: `phase`, `ball_x`, `ball_y`,
`ball_s`, beim zusammengeführten Ball zusätzlich `ball_from`) und liefert eine Zeile pro Frame mit
demselben Index:

| Spalte | Inhalt |
|---|---|
| `x`, `y` | Position 0 bis 1: gemessen oder geschätzt, `NaN` ohne Schätzung |
| `x_m`, `y_m` | dieselbe Position in Metern |
| `source` | `observed` (gemessen), `interpolated` (Lücke gefüllt), `missing` (keine Schätzung) |
| `gap_frames` | Länge der Lücke, in der der Frame liegt. `0` bei gemessen |
| `outlier` | Die Messung in diesem Frame wurde als Fehldetektion verworfen |
| `vx`, `vy` | Geschwindigkeit in m/s entlang x und y, aus den Positionen berechnet |
| `speed_ms` | Tempo in m/s, aus den Positionen berechnet |
| `speed_exported` | das exportierte `ball_s`, Einheit laut Metadaten |
| `in_pitch` | Ball innerhalb der Spielfeldlinien |
| `ball_from` | nur beim zusammengeführten Ball: woher die Sichtung stammt (`first`, `second`, `rejected`, `none`) |

Die Regeln werden unter `ball` in `config.yaml` eingestellt und gelten jeweils nur innerhalb einer
Phase. Über die Halbzeitpause wird nichts verbunden.

1. **Fehldetektion:** Ein einzelner Messpunkt, der schneller als `max_speed_ms` (40 m/s) von
   beiden Nachbarn wegspringt, während die Nachbarn zueinander passen, wird verworfen.
2. **Lücke:** Eine Lücke zwischen zwei Sichtungen von höchstens `max_gap_s` (5 s) wird auf der
   Geraden mit konstantem Tempo gefüllt. Längere Lücken und Lücken am Phasenrand bleiben
   `missing`.
3. **Geschwindigkeit:** berechnet aus den Positionen ±`velocity_window` (2) Frames.

**Lücken pro Halbzeit** (`main.py`, auf den gelieferten Rohdaten, also vor dem Füllen):

| Funktion | Ergebnis |
|---|---|
| `frames_without_ball(ball)` | pro Frame die Zahl der Frames in Folge ohne Ball, `0` wenn gesehen. Die Spitze jeder Zacke ist die Länge einer Lücke |
| `gap_stats(ball, config)` | Anzahl der Lücken, Median, Lücken über `max_gap_s`, **längste Lücke** in Frames und Sekunden mit Frame-Bereich |
| `plot_frames_without_ball(ball, config)` | Video-Frame gegen Frames ohne Ball, mit der Grenze `max_gap_s` und markierter längster Lücke |

`python -m src.ball_analysis` zeigt pro Phase die Anteile gemessen, gefüllt und fehlend, die
Verteilung der Lückenlängen und ob `ball_s` in km/h ist. Liegt das Verhältnis zur berechneten
Geschwindigkeit bei etwa 3,6, ist es km/h.

## 9. Umrechnungen

**Frame ↔ Zeit.** Alle Frames sind Frames des taktischen Videos `tactical/BAR-ATH_tactical.mp4`
(25 pro Sekunde). Die Spieluhr beginnt an jedem Anpfiff neu (`clock_start_min` in `config.yaml`):

| Frame | Videozeit | Spieluhr |
|---|---|---|
| 0 | 0:00:00 | – (vor dem Spiel) |
| 11.185 | 0:07:27 | **0:00**, Anpfiff 1. Halbzeit |
| 90.651 | 1:00:26 | 45+7:58, Ende 1. Halbzeit laut DATA.md |
| 90.652 – 113.741 | | – (Pause, dann vor dem Anpfiff) |
| 113.742 | 1:15:49 | **45:00**, Anpfiff 2. Halbzeit |
| 190.197 | 2:06:47 | 90+5:58, Abpfiff |

```bash
python -m src.phases --frame 20000     # Videozeit, Phase, Spieluhr eines Frames
python -m src.phases --clock 45+2:30   # Frame zu einer Spieluhr-Angabe (auch "23:10", "67:05", "90+1:00")
```

In Python: `video_time(frame, config)`, `match_clock(frame, config)`, `clock_to_frame("23:10", config)`
aus `src/phases.py`. Der Viewer zeigt beides im Titel.

| | |
|---|---|
| Video-Frame | Datei-Frame − 1 |
| Sekunden (Video) | Frame ÷ 25 |
| Spieluhr 1. Halbzeit | (Frame − 11.185) ÷ 25 Sekunden ab 0:00 |
| Spieluhr 2. Halbzeit | 45:00 + (Frame − 113.742) ÷ 25 Sekunden |
| Meter | `x` × 105,3, `y` × 68,0 |
| m/s | km/h ÷ 3,6 |

---

## 10. Video 2 (Broadcast) auf der taktischen Zeitachse

Bestimmt mit `python -m src.broadcast_sync` aus den Spielerpositionen beider Trackings. In der
1. Halbzeit passen 99 % der Proben, in der 2. Halbzeit 98 %. Die Steigung ist 1,0000, es gibt also
keine Streckung. Die Werte stehen **vorläufig von Hand** in `config.yaml` unter `broadcast_offsets`.
Vor der Abgabe muss die Pipeline sie selbst berechnen (siehe TODO dort).

| | Versatz (taktischer Frame = TV-Frame + Versatz) | Anpfiff im TV | Ende im TV |
|---|---|---|---|
| 1. Halbzeit | +4.014 Frames (2:40,56) | Frame 7.171 (0:04:46,84) | Frame 86.637 |
| 2. Halbzeit | +26.919 Frames (17:56,76) | Frame 86.823 (0:57:52,92) | Frame 163.278 (1:48:51,12) |

Die Halbzeitpause fehlt im TV-Video fast ganz: 7,4 s statt 15:24 min. Der Ball aus Video 2 zählt nur
dort, wo das TV-Bild eine gültige Homografie hat. Wiederholungen in der Totalen werden dabei nicht
erkannt.

**Zusammenführen (`balls["merged"]` in `main.py`, Einstellungen unter `merge` in `config.yaml`):**

1. Wo Video 1 den Ball sieht, gilt Video 1.
2. In Lücken von Video 1, die länger als `min_gap_s` (5 s) sind, wird die Ballposition aus Video 2
   übernommen, sofern Video 2 den Ball dort sieht. Kürzere Lücken füllt die Interpolation.
3. Prüfung gegen Wiederholungen (`check_players`): Ein TV-Frame wird nur verwendet, wenn seine
   Spieler zu den taktischen Spielern im selben Moment passen. Gemessen wird der typische Abstand
   (höchstens `max_player_rms_m` = 3 m, live sind es etwa 1 m), mit mindestens `min_players` (4)
   Spielern im Bild.

Die Spalte `ball_from` sagt pro Frame, woher der Ball kommt: `first` (Video 1), `second` (Video 2),
`rejected` (Video 2 hatte einen Ball, aber die Spieler passen nicht, also wohl eine Wiederholung)
oder `none`.

## 11. Tote Bälle (`src/dead_ball.py`, `dead` in `main.py`)

**Pro Frame** (`dead.indicators`): `crowd` gibt an, wie viele Spieler höchstens im Kreis von 10 m um
einen Spieler stehen, also an der dichtesten Stelle. `crowd_x` und `crowd_y` sagen, wo diese Stelle
liegt. `speed_ms` ist das Median-Tempo der Spieler.

**Ballungen** (`dead.episodes`): mindestens 12 Spieler im Kreis, mindestens 3 s lang. Die dichteste
hat ID 1.

**Lange Lücken** (`dead.gaps`): Stellen, an denen `data.ball` nach Zusammenführen und Interpolation
noch länger als 5 s fehlt. Sie sind in zeitlicher Reihenfolge nummeriert. Jede Lücke hat diese
Hinweise:

| Spalte | Hinweis auf toten Ball, wenn … |
|---|---|
| `speed_ms` | ≤ `slow_speed_ms` (1,5 m/s): Die Spieler gehen |
| `crowd_peak` und Grund „players crowd“ | Eine Ballung überschneidet sich mit der Lücke |
| `ball_before` | `out of play`, `touchline`, `goal line` oder `corner`: Der Ball ging ins Aus |
| `ball_after` | `corner`, `goal area`, `centre spot`, `penalty spot` oder `touchline`: Der Ball taucht an einem Standardpunkt auf |
| `verdict` | `dead` bei mindestens einem Hinweis, sonst `unclear` |

Ansehen im Viewer: `--episodes` / `--episode N` für die Ballungen, `--gaps` / `--gap N` für die
Lücken. Mit den Tasten `n` und `p` geht es weiter bzw. zurück.

## 12. Standardsituationen (`src/restarts.py`)

Ein Standard: Der Ball liegt mindestens `min_still_s` (1 s) still (Tempo ≤ `still_speed_ms`) und
wird dann gespielt. Der letzte stille Frame liegt in einer Halbzeit. Die Lage des Balls bestimmt
die Art, die Art das Team:

| Art | Ball | Team |
|---|---|---|
| `corner` | ≤ 2,5 m an einer Eckfahne | greift dieses Tor an |
| `goal kick` | im Torraum | verteidigt dieses Tor |
| `penalty` | ≤ 1,5 m am Elfmeterpunkt | greift dieses Tor an |
| `kickoff` | ≤ 1,5 m am Anstoßpunkt | des Ausführenden |
| `throw-in` | an der Seitenlinie | des Ausführenden |
| `keeper` | sonst, Torwart am nächsten | des Torwarts |
| `free kick` | sonst | des Ausführenden (auch ein im Spiel ruhender Ball) |
| `off pitch`, `goal line` | wartet hinter einer Linie | keins |

Ein ruhender Ball ist erst einmal nur ein Kandidat. Ein Standard (`verdict` = `restart`) ist er
nur, wenn er je Art lange genug ruht (`min_duration_s`, z. B. Einwurf 2 s, Freistoß 3 s), der
nächste Gegner Abstand hält (`opponents_back_m`, z. B. Einwurf 1,5 m, Freistoß 3 m; die Regeln
verlangen 2 m bzw. 9,15 m) und die Spieler langsam werden (mittleres Tempo über die Strecke
≤ `stoppage_speed_ms`, 2 m/s). Sonst `in play` mit Begründung in `why`: Zweikampf an der Linie,
Foul im laufenden Spiel, Ball kurz gestoppt. Außerdem `keeper holds` (Torwart hat den Ball, Spiel
läuft) und `waiting` (Ball liegt hinter einer Linie).

Einwürfe ohne sichtbaren Ball (`by` = `thrower`): ein Spieler steht mindestens 1 s still,
mindestens `thrower_behind_m` (0,3 m) hinter der Seitenlinie. Wo der Ball zu sehen ist, ist er
höchstens `ball_near_m` (5 m) von ihm entfernt, und nach `back_on_s` (3 s) ist der Spieler wieder
auf dem Feld. Dazu kommen dieselben Prüfungen wie oben. Nur dort, wo kein ruhender Ball dieselbe
Zeit schon abdeckt.

Ein Freistoß braucht außerdem eine Vorgeschichte: In den `before_s` (10 s), bevor der Ball zur
Ruhe kommt, war ein Gegner des Ausführenden höchstens `duel_m` (2 m) am Ball, also Foul oder
Zweikampf (Spalte `duel_m`: so nah kam der nächste Gegner). Hält ein Spieler im Aufbau den Ball,
bleiben die Gegner weg, dann `in play: no duel before`.

Ausführender = der nächste Spieler im Moment des Spielens (gehaltene Torhüter und Leute weiter als
5 m neben dem Feld zählen nicht). `opponent_m` ist der Abstand des nächsten Gegners, `agrees`
prüft bei den Arten, die das Tor entscheidet, ob der Ausführende zu diesem Team gehört.

Achtung: Beim Ausball selbst (Beginn der Unterbrechung) ist der nächste Spieler oft der, der den
Ball ins Aus gespielt hat, also der Gegner. Entscheidend ist der Moment der Ausführung.

---

## 13. Smart Tagging, geprüft (`src/smart_tagging.py`)

Metricas automatische Annotation (`<game>_tactical_smart_tagging.xml`, SportsCode): Intervalle
mit Code (z. B. `SET PIECES`, `BUILD UP`) und Tags (`Team`, `Type`, …). Zeiten sind Sekunden ab
dem ersten Frame des taktischen Videos, also Frame = round(Sekunden × 25). Die Tags sind
automatisch erzeugt, nicht gesichert: Es gibt falsch positive und falsch negative, und jeder
Einwurf hat Team N/A.

Jeder Standard-Tag wird zweimal gegen das Tracking geprüft:

- `ours`: ein ruhender Ball von uns derselben Art, gespielt höchstens `margin_s` (3 s) neben dem
  Intervall, mit Urteil und Team;
- `fit`: Wie stark ist das Merkmal der Art im Intervall? Verglichen mit allen gleich langen
  Momenten der Halbzeiten: der Anteil, den es übertrifft. Merkmale: Einwurf = Spieler an/hinter
  der Seitenlinie, Ecke = Spieler im Strafraum, Abstoß = Ball im Torraum, Anstoß = Spieler in der
  eigenen Hälfte, Freistoß = Spieler langsam. `stoppage` = wie langsam die Spieler sind.

Urteil: `confirmed` (unser Standard dieser Art, oder `fit` ≥ 0,9), `contradicted` (keiner von uns
und `fit` < 0,5), sonst `unclear`. Team: das des Tags, sonst unseres.

---

## 14. Das Spiel an den toten Bällen zerlegt (`src/segments.py`)

**Unterbrechungen** (`events`): erkannte Standards (Urteil `restart`) und lange Ballpausen mit
Urteil `dead`, zusammengeführt, wenn sie sich überlappen oder höchstens `merge_s` (1 s) auseinander
liegen. Die dichten Momente sind Indizien und bleiben draußen. `clear` = die Unterbrechung endet mit
einem Standard; `played` = der Frame, in dem der Ball gespielt wird.

**Ballbesitz danach** (`after_team`): in den `after_s` (3 s) ab `played` nur die Frames, in denen
ein Spieler höchstens `control_m` (2 m) am Ball ist; dort das Team des nächsten Spielers, per
Mehrheit. `after_share` = Anteil dieser Mehrheit, `control_frames` = wie viele solche Frames. Ohne
solche Frames gilt das Team des Standards. `agrees` = stimmt mit dem Team des Standards überein.

**Wo das Spiel stoppte** (`start`, `start_by`): vom ruhenden Ball zurück, höchstens
`max_lookback_s` (60 s) und nie über die vorige Unterbrechung hinaus:

1. `ball out`: der erste Frame, in dem der Ball mehr als `out_margin_m` (0,5 m) hinter einer Linie
   ist. Stücke weniger als `rejoin_s` (3 s) auseinander zählen als eins. Gilt nur, wenn der Ball
   höchstens `max_out_gap_s` (15 s) nach dem letzten Aus zur Ruhe kommt;
2. `players slow`: sonst, wenn die Spieler beim ruhenden Ball gehen (Median ≤ `slow_speed_ms`,
   1,5 m/s): wo sie damit anfingen, `reaction_s` (1 s) früher (der Pfiff);
3. `-`: sonst der ruhende Ball.

Der tote Ball reicht von `start` bis `played` (`dead_s`).

**Ballbesitz davor** (`before_team`): wie danach, in den `before_s` (3 s) vor `start`.
`last_touch` = Team des nächsten Spielers im letzten Frame mit Kontrolle davor. `out_check`: Bei
Einwurf, Ecke und Abstoß hat das andere Team den Ball ins Aus gespielt (`ok`), sonst `odd`.

**Lebende Abschnitte** (`live`): von `played` + 1 bis zum nächsten `start` − 1, mit dem Team am
Anfang (`team`) und am Ende (`end_team`).

---

## 15. Was **nicht** in den Daten steht

- **Wer den Ball hat:** Das ist die Aufgabe.
- **Ob der Ball im Spiel ist** (`D`): ebenfalls herzuleiten.
- **Ereignisse:** Pässe, Zweikämpfe, Fouls und Einwürfe sind nicht markiert.
- **Feste Spieler-Identitäten:** nur Spuren.
- **Ballhöhe:** nur 2D.
- **Bewegungsrichtung:** Nur der Betrag des Tempos steht drin. Die Richtung kommt aus den
  Positionsdifferenzen aufeinanderfolgender Frames.
