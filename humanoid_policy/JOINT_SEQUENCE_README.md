# Joint Sequence Player

Dieses Skript ermöglicht es, Gelenkbewegungen aus einer Textdatei Zeile für Zeile in Isaac Sim auszuführen, ähnlich wie im SpdrBot Beispiel.

## Schnellstart

**WICHTIG:** Diese Schritte müssen genau in dieser Reihenfolge ausgeführt werden!

```bash
# 1. Navigieren Sie zum IsaacLab Verzeichnis
cd C:\Users\2003n\IsaacLab

# 2. Aktivieren Sie die conda-Umgebung
..\env_isaaclab\Scripts\activate

# 3. Navigieren Sie zum Projektverzeichnis
cd C:\Users\2003n\projects\humanoid_tracked_legged_robot\policy\humanoid_legs_simlab\humanoid_policy

# 4. Starten Sie das Skript
python scripts/joint_sequence_player.py --file joint_sequence.txt --num_envs 1
```

Das Skript startet automatisch Isaac Sim und spielt die Bewegungen ab!

## Verwendung

### 1. Textdatei mit Gelenkpositionen erstellen

Erstellen Sie eine Textdatei (z.B. `joint_sequence.txt`) mit den gewünschten Gelenkpositionen. Jede Zeile enthält 8 Gelenkpositionen in Radiant.

**Format:**
- Leerzeichen-getrennt: `0.0 0.0 0.3491 -0.3491 0.0 0.0 -0.3491 0.3491`
- Komma-getrennt: `0.0, 0.0, 0.3491, -0.3491, 0.0, 0.0, -0.3491, 0.3491`
- Kommentare mit `#` werden ignoriert

**Gelenkreihenfolge:**
1. Revolute_11 - Hüfte Flexion links
2. Revolute_12 - Hüfte Abduktion links
3. Revolute_13 - Knie links
4. Revolute_14 - Knöchel links
5. Revolute_15 - Hüfte Flexion rechts
6. Revolute_16 - Hüfte Abduktion rechts
7. Revolute_17 - Knie rechts
8. Revolute_18 - Knöchel rechts

**Beispiel-Datei:**
```
# Initial pose (standing)
0.0 0.0 0.3491 -0.3491 0.0 0.0 -0.3491 0.3491

# Slight bend in left leg
0.2 0.0 0.5 -0.3 0.0 0.0 -0.3491 0.3491

# Return to initial
0.0 0.0 0.3491 -0.3491 0.0 0.0 -0.3491 0.3491
```

### 2. Skript ausführen

**WICHTIG:** Diese Schritte müssen **genau in dieser Reihenfolge** ausgeführt werden! Das Skript startet Isaac Sim automatisch. Sie müssen Isaac Sim nicht manuell öffnen!

#### Schritt-für-Schritt Anleitung:

```bash
# 1. Navigieren Sie zum IsaacLab Verzeichnis
cd C:\Users\2003n\IsaacLab

# 2. Aktivieren Sie die conda-Umgebung
..\env_isaaclab\Scripts\activate

# 3. Navigieren Sie zum Projektverzeichnis
cd C:\Users\2003n\projects\humanoid_tracked_legged_robot\policy\humanoid_legs_simlab\humanoid_policy

# 4. Starten Sie das Skript
python scripts/joint_sequence_player.py --file joint_sequence.txt --num_envs 1
```

#### Weitere Optionen:

```bash
# Mit Loop (wiederholt die Sequenz kontinuierlich)
python scripts/joint_sequence_player.py --file joint_sequence.txt --loop --num_envs 1

# Mit Anpassung der Verzögerung zwischen Frames (in Sekunden)
python scripts/joint_sequence_player.py --file joint_sequence.txt --delay 0.2 --num_envs 1

# Alles zusammen
python scripts/joint_sequence_player.py --file joint_sequence.txt --delay 0.2 --loop --num_envs 1
```

**Hinweis:** Stellen Sie sicher, dass Sie in der `env_isaaclab` conda-Umgebung sind (Sie sehen `(env_isaaclab)` vor dem Prompt) und im Verzeichnis `humanoid_policy` sind, bevor Sie den Befehl ausführen!

### 3. Kommandozeilen-Optionen

- `--file`: Pfad zur Textdatei mit Gelenkpositionen (Standard: `joint_sequence.txt`)
- `--delay`: Verzögerung zwischen Frames in Sekunden (Standard: 0.1)
- `--loop`: Sequenz kontinuierlich wiederholen
- `--num_envs`: Anzahl der Umgebungen (Standard: 1, für bessere Performance)
- `--task`: Task-Name (Standard: `Isaac-Humanoid-Policy-Direct-v0`)
- `--disable_fabric`: Fabric deaktivieren (falls Probleme auftreten)

## Beispiel-Sequenzen

Eine Beispiel-Datei `joint_sequence.txt` ist bereits im Projekt enthalten mit verschiedenen Bewegungsmustern:
- Initiale Pose
- Beugen des linken Beins
- Beugen des rechten Beins
- Beide Beine beugen
- Hüftflexion links und rechts

## Hinweise

- Alle Gelenkpositionen müssen in **Radiant** angegeben werden
- Die Simulation läuft mit 120 Hz, daher werden Bewegungen flüssig dargestellt
- Verwenden Sie `--num_envs 1` für bessere Performance beim Abspielen
- Drücken Sie `Ctrl+C` zum Beenden

## Fehlerbehebung

**Problem:** "Expected 8 joint values, got X"
- **Lösung:** Stellen Sie sicher, dass jede Zeile genau 8 Werte enthält

**Problem:** "Joint sequence file not found"
- **Lösung:** Überprüfen Sie den Pfad zur Datei mit `--file`

**Problem:** Bewegungen zu schnell/langsam
- **Lösung:** Passen Sie `--delay` an (größerer Wert = langsamere Bewegung)

**Problem:** "ModuleNotFoundError: No module named 'isaaclab'"
- **Lösung:** Stellen Sie sicher, dass Sie die conda-Umgebung aktiviert haben:
```bash
cd C:\Users\2003n\IsaacLab
..\env_isaaclab\Scripts\activate
cd C:\Users\2003n\projects\humanoid_tracked_legged_robot\policy\humanoid_legs_simlab\humanoid_policy
python scripts/joint_sequence_player.py --file joint_sequence.txt --num_envs 1
```

