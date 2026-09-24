import os
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import manual_kmeans
import ml_core

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evidence')
os.makedirs(OUT, exist_ok=True)


def write_b64(name, b64data):
    with open(os.path.join(OUT, name), 'wb') as f:
        f.write(base64.b64decode(b64data))


print('== Manual exercise plots ==')
state = manual_kmeans.run_simulation()
state = manual_kmeans.iteration_plots(state)
write_b64('01_initial_centroids.png', state['initial_plot'])
for it in state['iterations']:
    write_b64(f'02_iteration_{it["iteration"]}_clusters.png', it['plot'])
    print(f"  Iter {it['iteration']}: counts={it['counts']} WCSS={it['wcss']}")
write_b64('03_wcss_comparison.png', manual_kmeans.wcss_plot(state))

rows = manual_kmeans.distance_table(state)
pd.DataFrame(rows).to_csv(os.path.join(OUT, '04_distance_table_iter1.csv'), index=False)
print('  distance table saved')

summary_rows = []
for it in state['iterations']:
    summary_rows.append({
        'iteration': it['iteration'],
        'wcss': it['wcss'],
        'cluster_sizes': it['counts'],
        'centroids_before': [list(c) for c in it['centroids_before']],
        'centroids_after': [list(c) for c in it['centroids_after']],
    })
pd.DataFrame(summary_rows).to_json(os.path.join(OUT, '05_iterations_summary.json'), orient='records')
print('  iterations summary saved')

cent = pd.DataFrame(
    state['final_centroids'],
    columns=['distance_km', 'duration_min'])
cent.to_csv(os.path.join(OUT, '06_final_centroids.csv'), index=False)

print('== Clustering application ==')
app = ml_core.cluster_application()
write_b64('07_kmeans_clusters.png', app['plot'])
print(f"  silhouette={app['silhouette']}")
pd.DataFrame({
    'cluster': [0, 1, 2],
    'records': [app['summary_count'].get(i, 0) for i in range(3)],
    'centroid_distance_km': app['centroids_raw'][:, 0],
    'centroid_duration_min': app['centroids_raw'][:, 1],
}).to_csv(os.path.join(OUT, '08_kmeans_summary.csv'), index=False)
pd.DataFrame(app['sample']).to_csv(os.path.join(OUT, '09_kmeans_sample.csv'), index=False)

print('== Classification ==')
clf = ml_core.classification_application()
write_b64('10_confusion_matrices.png', clf['confusion_plot'])
for key in ('logistic', 'tree'):
    m = clf[key]
    print(f"  {m['name']}: acc={m['accuracy']} prec={m['precision']} rec={m['recall']} f1={m['f1']}")
    pd.DataFrame([m]).T.to_csv(os.path.join(OUT, f'11_{key}_metrics.csv'))
    pd.DataFrame(m['confusion']).to_csv(os.path.join(OUT, f'12_{key}_confusion.csv'), header=['pred_0', 'pred_1'], index=['act_0', 'act_1'])

print('== Done ==')
print('Evidence saved to:', OUT)