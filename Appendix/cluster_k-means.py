import os
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Input and output paths
input_files = [
    # Here all the examined files were included 
]
output_folder = r"C:\Users\kstat\Documents\Dissertation\Data\Methods\Method_2\Clustering\K-Means\K-Means_normalized"
os.makedirs(output_folder, exist_ok=True)

# Parameters for K-Means
cluster_range = [3, 10]  
random_state = 42
max_iter = 300
init_method = "k-means++"

# Function to load UMAP embeddings from a JSON file
def load_embeddings(file_path):
    print(f"Loading embeddings from: {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)
    embeddings = np.array([record["umap"] for record in data])  
    return data, embeddings

# Function to run t-SNE and save the plot
def run_tsne_and_save(embeddings, cluster_labels, output_path):
    print("Running t-SNE for visualization...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=40,
        learning_rate=200,
        n_iter=1000,
        early_exaggeration=24,
        metric='correlation'  # Match UMAP embedding metric
    )
    tsne_result = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, cmap="viridis", s=10, alpha=0.8)
    plt.colorbar(scatter, label="Cluster Labels")
    plt.title("t-SNE Visualization of Clustering")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"t-SNE plot saved to: {output_path}")
    plt.close()

# Main function to process each UMAP file and perform clustering
def process_umap_configuration(input_file, output_folder, n_clusters, random_state, max_iter, init_method):
    # Load embeddings from the specified input file
    try:
        data, embeddings = load_embeddings(input_file)

        print(f"Testing with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter, init=init_method)
        cluster_labels = kmeans.fit_predict(embeddings)

        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        print(f"Silhouette score for {n_clusters} clusters (file: {input_file}): {silhouette_avg:.4f}")

        # Compute silhouette coefficients
        silhouette_coeffs = silhouette_samples(embeddings, cluster_labels)
        for i, record in enumerate(data):
            record["cluster_label"] = int(cluster_labels[i])
            record["silhouette_coefficient"] = float(silhouette_coeffs[i])

        return silhouette_avg, data, embeddings, cluster_labels

    except KeyError as e:
        print(f"Error loading data from {input_file}: {e}")
        return -1, None, None, None

# Loop through each input file
for input_file in input_files:
    # Test for each number of clusters in the range (5, 6)
    for n_clusters in cluster_range:
        silhouette_avg, data, embeddings, cluster_labels = process_umap_configuration(
            input_file=input_file,
            output_folder=output_folder,
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            init_method=init_method
        )

        # Save the clustered data to JSON
        clustered_output_path = os.path.join(
            output_folder,
            f"clustered_embeddings_{input_file.split(os.sep)[-1].replace('.json', '')}_{n_clusters}clusters.json"
        )
        with open(clustered_output_path, "w") as f:
            json.dump({"mean_silhouette_score": silhouette_avg, "records": data}, f, indent=4)
        print(f"Clustered data saved to: {clustered_output_path}")

        # Print silhouette score
        print(f"Silhouette score for file {input_file} with {n_clusters} clusters: {silhouette_avg:.4f}")

        # Run t-SNE and save the plot
        tsne_output_path = os.path.join(
            output_folder,
            f"tsne_plot_{input_file.split(os.sep)[-1].replace('.json', '')}_{n_clusters}clusters.png"
        )
        run_tsne_and_save(embeddings, cluster_labels, tsne_output_path)
