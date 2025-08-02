from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    file = request.files['file']
    df = pd.read_csv(file)

    # Select features for clustering
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Save cluster plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='Set2')
    plt.title('Customer Segments')
    plot_path = os.path.join('static', 'cluster_plot.png')
    plt.savefig(plot_path)
    plt.close()

    return render_template('result.html', tables=[df.to_html(classes='data')], image='cluster_plot.png')

if __name__ == '__main__':
    app.run(debug=True)