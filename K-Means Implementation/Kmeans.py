# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_data(csv_file):
    data = pd.read_csv(csv_file)
    print("Dataset Preview:\n", data.head())  # Debugging line
    
    if data.shape[1] < 2:
        raise ValueError("Dataset must have at least two columns for clustering visualization.")
    
    X = data.iloc[:, :2].values  # Ensure selecting only the first two columns
    return X

def train_kmeans(X, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(X)
    return model

def evaluate_kmeans(X, model):
    silhouette_avg = silhouette_score(X, model.labels_)
    print(f"Silhouette Score: {silhouette_avg:.2f}")

def visualize_clusters(X, model):
    if X.shape[1] != 2:
        print("Error: X does not have exactly 2 features. Cannot visualize.")
        return
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap='viridis', marker='o', edgecolor='k', s=100)
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=300, c='red', marker='x', label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def main():
    csv_file = 'kmeans_data.csv'  # Replace with your CSV file path
    X = load_data(csv_file)
    
    n_clusters = 3  # Number of clusters
    model = train_kmeans(X, n_clusters=n_clusters)
    
    evaluate_kmeans(X, model)
    
    visualize_clusters(X, model)

if __name__ == "__main__":
    main()
