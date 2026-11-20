# XoNet — Analyse de Sentiment en Fongbé

**XoNet** est un prototype de classification de sentiment pour des phrases en **fon (fongbé)** selon 3 classes : **Positif (2)**, **Neutre (0)** et **Négatif (1)**.

## Vue d'ensemble du pipeline

1. **Corpus & Annotation** : 87 126 phrases en fongbé issues de textes parallèles et corpus collectés (`data/corpus_final.csv` est la référence annotée, `data/corpus_clean.csv` en extrait uniquement `fon`, `fr`, `sentiment_final`).
   - Partitionnement stratifié constant (`random_state=42`) : 70 % Train (60 988) / 15 % Validation (13 069) / 15 % Test (13 069).
2. **Modèles de Classification** :
   - **v1 — Soft Voting (MNB + Random Forest)** (`notebooks/train_model.ipynb`) : TF-IDF caractères `char_wb (3,5)` + SMOTE + Soft Voting + pondération des seuils.
   - **v2 — LinearSVC Calibré** (`src/training/train_sentiment_v2.py`) : TF-IDF `char_wb (2,6)` (80k) + SVM linéaire calibré + ajustement des seuils sur F1 macro.
   - **v3 Tuned — SVM Hybride Char+Word** (`src/training/tune_sentiment_v3.py`) : combinaison TF-IDF caractères `(2,6)` + mots `(1,5)` (150k features) + LinearSVC ($C=0.5$) calibré + seuils optimisés.
3. **Interface Web** (`app.py` + `web/index.html`) : application Flask permettant d'analyser des phrases en fongbé en temps réel avec affichage du sentiment (Positif / Neutre / Négatif), du score par classe et sélection du modèle (**SVM Hybride v3**, **Soft Voting**, **LinearSVC v2**).

## Comparatif des modèles (Jeu de Test — 13 069 phrases)

| Modèle | Script / Notebook | Accuracy | F1 Macro | F1 Neutre / Négatif / Positif |
| :--- | :--- | :---: | :---: | :---: |
| **Soft Voting (MNB + RF)** | `notebooks/train_model.ipynb` | 71.90 % | 65.20 % | 0.80 / 0.59 / 0.57 |
| **LinearSVC Calibré (v2)** | `src/training/train_sentiment_v2.py` | 72.28 % | 66.59 % | 0.80 / 0.62 / 0.58 |
| **SVM Hybride Char+Word (v3 Tuned)** | `src/training/tune_sentiment_v3.py` | **75.39 %** | **70.51 %** | **0.82 / 0.68 / 0.62** |

## Lancer l'application Web

```powershell
pip install -e .
python app.py
```

Ouvrir ensuite `http://localhost:5000`. Par défaut, l'application charge **`sentiment_pipeline_v3_tuned.joblib`** (et précharge `voting_model.joblib` pour permettre la bascule instantanée dans l'interface). Endpoints exposés : `GET /api/models` et `POST /api/predict`.

Variables d'environnement optionnelles :

```powershell
# Forcer le mode v2 par défaut :
$env:XONET_PIPELINE_FILENAME="sentiment_pipeline_v2.joblib"
python app.py

# Forcer le mode Soft Voting (legacy) par défaut :
$env:XONET_PIPELINE_FILENAME=""
python app.py
```

## Ré-entraînement des pipelines ML

### Pipeline v2

```bash
python -m src.training.train_sentiment_v2
# Ou avec LightGBM/XGBoost :
python -m src.training.train_sentiment_v2 --include-boosting
```

Sorties créées dans `models/` : `sentiment_pipeline_v2.joblib`, `metrics_v2.json`, `classification_report_v2.txt`. La sélection du modèle se fait sur `F1 macro` en validation.

### Pipeline v3 (Hybride Char + Word SVM)

```bash
python -m src.training.train_sentiment_v3
python -m src.training.tune_sentiment_v3
```

Sorties créées dans `models/` : `sentiment_pipeline_v3_tuned.joblib`, `metrics_v3_tuned.json`, `classification_report_v3_tuned.txt`.
