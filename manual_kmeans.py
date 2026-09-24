import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(SCRIPT_DIR, 'data', 'flights_manual_100.csv')

INITIAL_CENTROIDS = np.array([
    [400.0, 55.0],
    [1600.0, 165.0],
    [2700.0, 280.0],
], dtype=float)

PALETTE = ['#e63946', '#2a9d8f', '#457b9d']


def polar_df():
    df = pd.read_csv(DATA)
    return df


def scatter(ax, X, colors, centroids, title, annotate_centroids=True):
    ax.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.7, edgecolors='white',
               linewidths=0.4, s=42, zorder=2)
    if annotate_centroids:
        ax.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=210,
                   c='black', edgecolors='white', linewidths=1.2, zorder=3, label='Centroid')
        for i, (cx, cy) in enumerate(centroids):
            ax.annotate(f'C{i+1}', (cx, cy), textcoords='offset points',
                        xytext=(0, -12), ha='center', fontsize=9, fontweight='bold', color='black')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Duration (min)')
    ax.set_title(title)
    ax.grid(alpha=0.3)


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def euclidean(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2)))


def run_simulation():
    df = polar_df()
    X = df[['distance_km', 'duration_min']].values.astype(float)

    initial_plot = fig_to_b64(_initial_plot(X))

    centroids = INITIAL_CENTROIDS.copy()
    iterations = []

    for it in range(1, 4):
        d0 = np.array([euclidean(p, centroids[0]) for p in X])
        d1 = np.array([euclidean(p, centroids[1]) for p in X])
        d2 = np.array([euclidean(p, centroids[2]) for p in X])
        dists = np.column_stack([d0, d1, d2])
        assignments = np.argmin(dists, axis=1)

        wcss = 0.0
        for c in range(3):
            wcss += float(np.sum(dists[assignments == c, c] ** 2))

        new_centroids = np.zeros_like(centroids)
        for c in range(3):
            mask = assignments == c
            if np.any(mask):
                new_centroids[c] = X[mask].mean(axis=0)
            else:
                new_centroids[c] = centroids[c]

        prev = centroids.copy()

        iterations.append({
            'iteration': it,
            'distances': np.round(dists, 2),
            'assignments': assignments,
            'wcss': round(wcss, 2),
            'centroids_before': prev.round(2),
            'centroids_after': new_centroids.round(2),
            'counts': [int(np.sum(assignments == c)) for c in range(3)],
            'plot': None,
        })

        centroids = new_centroids

    return {
        'df': df,
        'X': X,
        'initial_centroids': INITIAL_CENTROIDS.round(2),
        'initial_plot': initial_plot,
        'iterations': iterations,
        'final_centroids': centroids.round(2),
    }


def _initial_plot(X):
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter(ax, X, '#94a3b8', INITIAL_CENTROIDS, 'Flights dataset with initial centroids',
            annotate_centroids=True)
    ax.legend(loc='upper left')
    return fig


def iteration_plots(state):
    X = state['X']
    for it in state['iterations']:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = [PALETTE[a] for a in it['assignments']]
        scatter(ax, X, colors, it['centroids_after'],
                f'Iteration {it["iteration"]} - clusters and updated centroids')
        ax.legend(loc='upper left')
        it['plot'] = fig_to_b64(fig)
    return state


def wcss_plot(state):
    wcss = [it['wcss'] for it in state['iterations']]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([f'Iter {i}' for i in range(1, 4)], wcss, color=PALETTE)
    for b, v in zip(bars, wcss):
        ax.text(b.get_x() + b.get_width() / 2, v + 20, f'{v:,.0f}',
                ha='center', fontweight='bold')
    ax.set_ylabel('WCSS')
    ax.set_title('Within-cluster variance (WCSS) per iteration')
    ax.grid(axis='y', alpha=0.3)
    return fig_to_b64(fig)


def distance_table(state, n_rows=8):
    df = state['df'].head(n_rows).copy()
    it = state['iterations'][0]
    for c in range(3):
        df[f'd_{c+1}'] = it['distances'][:n_rows, c]
    df['assigned_cluster'] = [a + 1 for a in it['assignments'][:n_rows]]
    return df.round(2).to_dict('records')


if __name__ == '__main__':
    state = run_simulation()
    state = iteration_plots(state)
    for it in state['iterations']:
        print(f"--- Iteration {it['iteration']} ---")
        print('counts:', it['counts'], 'WCSS:', it['wcss'])
        print('centroids before:', it['centroids_before'].tolist())
        print('centroids after:', it['centroids_after'].tolist())
    print('final centroids:', state['final_centroids'].tolist())