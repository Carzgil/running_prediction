import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


def perform_clustering(data):
    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Calculate scores for different numbers of clusters
    max_clusters = min(10, len(data)-1)  # Don't try more clusters than samples
    silhouette_scores = []

    for k in range(2, max_clusters+1):
        print(f"KMeans, k={k}")
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))

    # Find optimal k using silhouette score
    silhouette_k = np.argmax(silhouette_scores) + 2

    # Perform final clustering with optimal k
    final_kmeans = KMeans(n_clusters=silhouette_k, random_state=42)
    clusters = final_kmeans.fit_predict(scaled_data)

    return clusters


def perform_clustering_dbscan(data):
    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Start with default parameters
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    clusters = dbscan.fit_predict(scaled_data)

    # Number of clusters (excluding noise points)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    print(f"\nNumber of clusters found: {n_clusters}")
    print(f"Number of noise points: {(clusters == -1).sum()}")

    return clusters


def perform_clustering_agglomerative(data):
    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Try different numbers of clusters
    max_clusters = min(10, len(data)-1)
    silhouette_scores = []

    for k in range(2, max_clusters+1):
        print(f"Agglomerative, k={k}")
        clustering = AgglomerativeClustering(n_clusters=k)
        labels = clustering.fit_predict(scaled_data)
        silhouette_scores.append(silhouette_score(scaled_data, labels))

    # Find optimal k using silhouette score
    optimal_k = np.argmax(silhouette_scores) + 2

    # Perform final clustering with optimal k
    final_clustering = AgglomerativeClustering(n_clusters=optimal_k)
    clusters = final_clustering.fit_predict(scaled_data)

    print(f"\nNumber of clusters found: {optimal_k}")

    return clusters


def visualize_clusters(data, clusters, output_file):
    # Reduce dimensionality to 2D for visualization
    # pca = PCA(n_components=2)
    # reduced_data = pca.fit_transform(data)

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data.iloc[:, 0], data.iloc[:, 1],
                          c=clusters, cmap='viridis')

    # Add labels for each point
    # for i, file_name in enumerate(file_names):
    # plt.annotate(file_name, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

    plt.colorbar(scatter)
    plt.title('Document Clusters')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.savefig(output_file)
    plt.close()


def main():
    # Load the data
    daily_data = pd.read_csv('data/day_approach_maskedID_timeseries.csv')
    data = pd.read_csv('clustering/boids_positions.csv')
    data = data.drop(columns=['Athlete ID'])

    # Define clustering methods
    clustering_methods = [
        ('KMeans', perform_clustering),
        ('DBSCAN', perform_clustering_dbscan),
        ('Agglomerative', perform_clustering_agglomerative)
    ]

    # Iterate through each clustering method
    for method_name, clustering_func in clustering_methods:
        print(f"\n{'-'*20}\nPerforming {method_name} Clustering\n{'-'*20}")

        # Perform clustering
        clusters = clustering_func(data)

        if (method_name == 'DBSCAN'):
            # get cluster with most entries
            unique_clusters, counts = np.unique(clusters, return_counts=True)
            most_common_cluster = unique_clusters[np.argmax(counts)]

            # Get indices of points in the most common cluster
            cluster_indices = np.where(clusters == most_common_cluster)[0]

            # Get the data points for athletes with ids in cluster_indices
            filtered_data = daily_data[daily_data['Athlete ID'].isin(
                cluster_indices)]

            # save to csv
            filtered_data.to_csv(
                f'clustering/filtered_data_without_outliers.csv', index=False)

        # Count documents in each cluster
        unique_clusters = np.unique(clusters)
        print("\nCluster Statistics:")
        for cluster in unique_clusters:
            count = np.sum(clusters == cluster)
            print(f"Cluster {cluster}: {count} entries")

        # Visualize the clusters
        output_file = f'clustering_results_flocking_{method_name.lower()}.png'
        visualize_clusters(data, clusters, output_file)
        print(f"\nVisualization saved as '{output_file}'")


if __name__ == "__main__":
    main()
