import os
from flask import Flask, render_template
import manual_kmeans
import ml_core

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/concepts')
def concepts():
    return render_template('concepts.html')


@app.route('/manual-exercise')
def manual_exercise():
    state = manual_kmeans.run_simulation()
    state = manual_kmeans.iteration_plots(state)
    wcss_plot = manual_kmeans.wcss_plot(state)

    iterations = []
    for it in state['iterations']:
        iterations.append({
            'iteration': it['iteration'],
            'wcss': f"{it['wcss']:,.2f}",
            'counts': it['counts'],
            'centroids_before': [list(c) for c in it['centroids_before']],
            'centroids_after': [list(c) for c in it['centroids_after']],
            'plot': it['plot'],
        })

    initial = [tuple(c) for c in state['initial_centroids']]
    final = [tuple(c) for c in state['final_centroids']]
    distance_rows = manual_kmeans.distance_table(state)

    return render_template('manual.html',
                           initial_centroids=initial,
                           final_centroids=final,
                           initial_plot=state['initial_plot'],
                           iterations=iterations,
                           wcss_plot=wcss_plot,
                           distance_rows=distance_rows,
                           n_rows=len(manual_kmeans.polar_df()))


@app.route('/clustering-application')
def clustering_application():
    result = ml_core.cluster_application()
    return render_template('application.html', **result)


@app.route('/classification')
def classification():
    result = ml_core.classification_application()
    return render_template('classification.html', **result)


if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=debug)