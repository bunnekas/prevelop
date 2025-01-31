# PrEvelOp

PrEvelOp ist ein modular aufgebautes Python-Framework zur automatisierten Datenaufbereitung, explorativen Analyse, Clusterbildung und Evaluierung, das speziell für gemischte Datentypen wie geometrische und prozessbezogene Merkmale entwickelt wurde.

---

## Installation

### Schritt 1: Repository klonen

Klone das Repository und wechsel in das Projektverzeichnis:

```bash
git clone <repository-url>
cd prevelop
```
### Schritt 2: Virtuelle Umgebung einrichten

Erstelle und aktiviere eine virtuelle Umgebung, um die Abhängigkeiten zu isolieren:

```bash
# Virtuelle Umgebung erstellen
python3 -m venv prevelop-env

# Virtuelle Umgebung aktivieren
# Auf Linux/MacOS:
source prevelop-env/bin/activate

# Auf Windows:
prevelop-env\Scripts\activate
```

### Schritt 3: Abhängigkeiten installieren

Installiere die benötigten Abhängigkeiten aus der Datei `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Schritt 4: PrEvelOp im Entwicklungsmodus installieren

Installiere das Tool im Entwicklungsmodus, um Änderungen am Code direkt wirksam zu machen:

```bash
pip install -e .
```

### Schritt 5: Installation überprüfen

Überprüfe, ob die Installation erfolgreich war:

```bash
# Python-Shell starten
python


# Teste, ob das Tool importiert werden kann
>>> import prevelop
>>> print(prevelop.__version__)
```