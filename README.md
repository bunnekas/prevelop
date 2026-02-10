# PrEvelOp

PrEvelOp ist ein modular aufgebautes Python-Framework zur automatisierten Datenaufbereitung, explorativen Analyse, Clusterbildung und Evaluierung, das speziell für gemischte Datentypen wie geometrische und prozessbezogene Merkmale entwickelt wurde.

Das Framework bietet eine interaktive **Streamlit-Dashboard** für die visuelle Analyse und Clusterbildung sowie eine Python-API für die Verwendung in Notebooks und Skripten.

---

## Features

- **Datenaufbereitung**: Laden, Aggregieren und Vorverarbeiten von gemischten Datentypen (numerisch und kategorisch)
- **Explorative Analyse**: Boxplots, Verteilungen, Korrelations-Heatmaps, t-SNE, Ausreißererkennung
- **Clustering**: Agglomeratives Clustering, K-Medoids und HDBSCAN mit Gower-Distanz für gemischte Daten
- **Evaluierung**: Silhouette Score, Dunn Index, Davies-Bouldin Index, Calinski-Harabasz Score, Feature Importance
- **Dashboard**: Interaktive Streamlit-Oberfläche mit Plotly-Visualisierungen

---

## Installation

### Schritt 1: Repository klonen

```bash
git clone <repository-url>
cd prevelop
```

### Schritt 2: Abhängigkeiten installieren

Mit [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Für das Dashboard zusätzlich:

```bash
uv sync --extra dashboard
```

### Schritt 3: Installation überprüfen

```bash
uv run python -c "import prevelop; print(prevelop.__version__)"
```

---

## Verwendung

### Dashboard starten

```bash
uv run streamlit run app.py
```

Das Dashboard bietet vier Seiten:

| Seite | Beschreibung |
|---|---|
| **Overview** | Datensatz-Übersicht, deskriptive Statistiken |
| **Exploration** | Verteilungen, Boxplots, Korrelation, t-SNE |
| **Clustering** | Algorithmus-Auswahl, Parametereinstellung, Visualisierung |
| **Evaluation** | Metriken, Cluster-Statistiken, Feature Importance |

### Notebook verwenden

Siehe `notebooks/exploration.ipynb` für ein Anwendungsbeispiel mit der Python-API.

```python
from prevelop.data import generate_toy_dataset
from prevelop.preparation import preprocessing
from prevelop.clustering import gower_distance, agglomerative_clustering
from prevelop.evaluation import evaluate_clustering

# Daten laden
data, num_columns, cat_columns = generate_toy_dataset(n_samples=300)

# Vorverarbeitung
data_preprocessed = preprocessing(data, num_columns, cat_columns)

# Clustering
distance = gower_distance(data_preprocessed)
labels = agglomerative_clustering(distance, nr_cluster=5)

# Evaluierung
scores = evaluate_clustering(data_preprocessed.values, labels)
```

---

## Projektstruktur

```
prevelop/
├── app.py                      # Streamlit Dashboard
├── pyproject.toml              # Projektdefinition und Abhängigkeiten
├── README.md
├── notebooks/
│   └── exploration.ipynb       # Anwendungsbeispiel
└── src/
    └── prevelop/
        ├── __init__.py
        ├── data.py             # Synthetische Daten
        ├── preparation.py      # Datenaufbereitung
        ├── exploration.py      # Explorative Analyse
        ├── clustering.py       # Clustering-Algorithmen
        └── evaluation.py       # Evaluierung
```

---

## Pipeline

```
Daten laden → Vorverarbeitung → Gower-Distanz → Clustering → Evaluierung
     │              │                │              │             │
  load_data    preprocessing   gower_distance   agglo/kmed/   evaluate_
  (CSV/Excel)  (Scale+Encode)  (Mixed Types)    hdbscan      clustering
```
