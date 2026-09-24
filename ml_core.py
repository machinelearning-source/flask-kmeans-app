import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.inspection import permutation_importance

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_MAIN = os.path.join(SCRIPT_DIR, 'data', 'flights_dataset.csv')


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def load_main():
    return pd.read_csv(DATA_MAIN)


def cluster_application(recompute=False):
    df = load_main()
    feats = ['distance_km', 'duration_min']
    X_raw = df[feats].values.astype(float)
    X = X_raw.copy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(Xs)
    df['Cluster'] = labels

    sil = silhouette_score(Xs, labels)
    centers_scaled = kmeans.cluster_centers_
    centers_raw = scaler.inverse_transform(centers_scaled)

    summary = df.groupby('Cluster').size().to_dict()

    plot = _kmeans_plot(X_raw, labels, centers_raw, df, feats, sil)

    return {
        'sample': df.head(15).to_dict('records'),
        'summary_count': summary,
        'centroids_raw': centers_raw,
        'silhouette': round(sil, 4),
        'plot': plot,
        'feature_ranges': {
            'distance': (round(df['distance_km'].min(), 1), round(df['distance_km'].max(), 1)),
            'duration': (round(df['duration_min'].min(), 1), round(df['duration_min'].max(), 1)),
        },
    }


def _kmeans_plot(X, labels, centers_raw, df, feats, sil):
    palette = ['#e63946', '#2a9d8f', '#457b9d']
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for c in sorted(set(labels)):
        mask = labels == c
        ax.scatter(X[mask, 0], X[mask, 1], c=palette[c], alpha=0.55,
                   edgecolors='white', linewidths=0.3, s=18, label=f'Cluster {c}')
    ax.scatter(centers_raw[:, 0], centers_raw[:, 1], marker='X', s=230,
               c='black', edgecolors='white', linewidths=1.3, label='Centroids')
    for i, (cx, cy) in enumerate(centers_raw):
        ax.annotate(f'C{i}', (cx, cy), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontweight='bold')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Duration (min)')
    ax.set_title(f'K-Means clustering of flights (silhouette = {sil:.3f})')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')
    return fig_to_b64(fig)


def classification_application():
    df = load_main()
    feats = ['distance_km', 'duration_min']
    X = df[feats].values.astype(float)
    y = df['delayed'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)

    dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)

    results = {
        'logistic': {
            'name': 'Logistic Regression',
            'accuracy': round(accuracy_score(y_test, lr_pred), 4),
            'precision': round(precision_score(y_test, lr_pred), 4),
            'recall': round(recall_score(y_test, lr_pred), 4),
            'f1': round(f1_score(y_test, lr_pred), 4),
            'coefs': [round(c, 4) for c in lr.coef_[0]],
            'intercept': round(lr.intercept_[0], 4),
            'confusion': confusion_matrix(y_test, lr_pred).tolist(),
        },
        'tree': {
            'name': 'Decision Tree',
            'accuracy': round(accuracy_score(y_test, dt_pred), 4),
            'precision': round(precision_score(y_test, dt_pred), 4),
            'recall': round(recall_score(y_test, dt_pred), 4),
            'f1': round(f1_score(y_test, dt_pred), 4),
            'depth': dt.get_depth(),
            'confusion': confusion_matrix(y_test, dt_pred).tolist(),
            'importances': [round(i, 4) for i in dt.feature_importances_],
        },
        'features': feats,
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'pos_rate': round(float(y.mean()), 4),
        'confusion_plot': _confusion_plot(lr_pred, dt_pred, y_test),
    }
    return results


def _confusion_plot(lr_pred, dt_pred, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, preds, title in [(axes[0], lr_pred, 'Logistic Regression'),
                             (axes[1], dt_pred, 'Decision Tree')]:
        cm = confusion_matrix(y_test, preds)
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1], ['On-time (0)', 'Delayed (1)'])
        ax.set_yticks([0, 1], ['On-time (0)', 'Delayed (1)'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha='center', va='center', fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle('Confusion matrices')
    fig.tight_layout()
    return fig_to_b64(fig)