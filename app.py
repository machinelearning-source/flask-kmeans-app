import os
from flask import Flask, render_template
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Prevents GUI errors on headless servers like Render
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

app = Flask(__name__)

# Helper function to generate mockup production data
def get_production_data():
    np.random.seed(42)
    # Generate 1000 records: Annual Income (k$) and Spending Score (1-100)
    cluster1 = np.random.normal(loc=[30, 20], scale=[5, 5], size=(250, 2))
    cluster2 = np.random.normal(loc=[60, 50], scale=[8, 8], size=(400, 2))
    cluster3 = np.random.normal(loc=[90, 80], scale=[6, 6], size=(350, 2))
    
    data = np.vstack([cluster1, cluster2, cluster3])
    data = np.clip(data, 10, 100)
    df = pd.DataFrame(data, columns=['Annual_Income', 'Spending_Score'])
    return df

@app.route('/')
@app.route('/concepts')
def concepts():
    return render_template('concepts.html')

@app.route('/manual-exercise')
def manual_exercise():
    # Hardcoded values tracking the manual mathematical simulation from Part 1
    iterations_data = [
        {"iteration": 1, "wcss": 15420.5, "centroids": [[35.2, 45.1], [55.8, 62.3], [22.1, 15.4]]},
        {"iteration": 2, "wcss": 9840.2, "centroids": [[32.1, 48.9], [58.4, 60.1], [24.3, 18.2]]},
        {"iteration": 3, "wcss": 6120.8, "centroids": [[30.5, 52.3], [61.2, 58.4], [25.1, 19.5]]}
    ]
    return render_template('manual.html', iterations=iterations_data)

@app.route('/clustering-application')
def clustering_application():
    df = get_production_data()
    
    # Process K-Means using Scikit-learn
    X = df[['Annual_Income', 'Spending_Score']]
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    
    # Metrics
    sil = silhouette_score(X, df['Cluster'])
    centroids = kmeans.cluster_centers_
    summary = df['Cluster'].value_counts().to_dict()
    
    # Structure data for HTML tables
    records_sample = df.head(15).to_dict(orient='records')
    
    # Dynamic scatter plot generation saved directly to static folder
    plt.figure(figsize=(6, 4))
    colors = ['#FF5733', '#33FF57', '#3357FF']
    for i in range(3):
        cluster_set = df[df['Cluster'] == i]
        plt.scatter(cluster_set['Annual_Income'], cluster_set['Spending_Score'], c=colors[i], label=f'Cluster {i}', alpha=0.6)
    
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=150, label='Centroids')
    plt.title('Production Clustering Results')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.tight_layout()
    
    # Ensure static directory exists
    os.makedirs(os.path.join('static', 'img'), exist_ok=True)
    plot_path = os.path.join('static', 'img', 'production_chart.png')
    plt.savefig(plot_path)
    plt.close()
    
    return render_template('application.html', 
                           sample=records_sample, 
                           silhouette=round(sil, 4), 
                           centroids=centroids, 
                           summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
