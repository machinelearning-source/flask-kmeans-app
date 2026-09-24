# Flight Analytics — K-Means Clustering & Binary Classification

Flask web application built for the **Machine Learning** course (Semester 6, Systems Engineering, Universidad de Cundinamarca).

The project analyses **flight operations** with two complementary machine learning tasks:

1. **Unsupervised Learning (K-Means):** groups 1,000 flights into clusters based on `distance_km` and `duration_min`, discovering short-, medium- and long-haul segments.
2. **Supervised Learning (Binary target):** predicts whether a flight is `delayed` (1) or on time (0) using Logistic Regression and a Decision Tree, with accuracy, precision, recall, F1 and confusion matrices.

## Registered topic (Microsoft Teams)

> **Segmentation of flights by distance and duration, with binary prediction of delay.** Unsupervised entry.

## Dataset

- `data/flights_dataset.csv` — 1,000 records, numerical variables `distance_km` and `duration_min`, plus binary target `delayed`.
- `data/flights_manual_100.csv` — 100 records used for the manual K-Means simulation (3 iterations).
- Generated deterministically (seed 42) with `generate_flights_dataset.py`: short, medium and long-haul flights plus a realistic delay signal (18% → 55% → 82% in larger flights).

## Features

| Route | Section |
|---|---|
| `/` | Home |
| `/concepts` | Unsupervised learning, clustering, K-Means, centroids & iterations |
| `/manual-exercise` | Part 1: manual 3-iteration simulation (distances, assignments, centroids, WCSS, plots) |
| `/clustering-application` | Part 2: K-Means with Scikit-Learn, silhouette score, centroids, scatter plot |
| `/classification` | Binary target: flight delay prediction (LogReg vs Decision Tree) |

## Manual exercise highlights

- Initial centroids chosen to cover the data range.
- Euclidean distance computed for every record to each of the 3 centroids.
- Records assigned to the nearest centroid; centroids recomputed as cluster averages.
- WCSS decreases each iteration: proof of convergence.
- Final centroids (dist km, duration min): approx. (392, 50), (1499, 141), (2613, 236).

## Scikit-Learn configuration

```python
KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
```

Features are standardized with `StandardScaler` before clustering and before Logistic Regression. Silhouette score ≈ 0.69 (well-separated clusters).

## Local run

```bash
pip install -r requirements.txt
python generate_flights_dataset.py   # optional, data included
python app.py                        # http://127.0.0.1:5000
```

## Deployment (Render)

1. Push this repository to GitHub.
2. In Render: **New → Web Service → Connect the GitHub repo**.
3. Settings: runtime **Python 3**, build command `pip install -r requirements.txt`, start command `web: gunicorn app:app` (from `Procfile`).
4. Deploy. The app reads `PORT` automatically.

## Authors

- Luis Alberto Rebolledo Ariza — Universidad de Cundinamarca, Machine Learning, Semestre 6.