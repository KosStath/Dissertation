import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# Set paths
umap_dir = r'C:\Users\kstat\Documents\Dissertation\Data\Methods\Method_2\UMAP\UMAP_embeddings'
output_dir = r'C:\Users\kstat\Documents\Dissertation\Data\Methods\Method_2\Clustering\DBSCAN\DBSCAN_normalized'
os.makedirs(output_dir, exist_ok=True)

# DBSCAN parameter ranges
eps_values = np.arange(0.2, 0.9, 0.1)  
min_samples_values = range(2, 9)  
distance_metrics = ['cosine', 'correlation', 'euclidean']  

# Initialize variables to track the best configuration
best_silhouette_score = -1  
best_eps = None
best_min_samples = None
best_metric = None
best_umap_file = None
best_labels = None
best_data = None

# Iterate through all UMAP files
print("Loading UMAP configurations...")
umap_files = [f for f in os.listdir(umap_dir) if f.endswith('.json')]

for umap_file in umap_files:
    umap_path = os.path.join(umap_dir, umap_file)
    print(f"Processing UMAP configuration from: {umap_file}")

    
    with open(umap_path, 'r') as f:
        data = json.load(f)
    embeddings = np.array([record['umap_3d'] for record in data])

    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Test DBSCAN with parameter combinations
    for eps in eps_values:
        for min_samples in min_samples_values:
            for metric in distance_metrics:
                print(f"Running DBSCAN with eps={eps}, min_samples={min_samples}, metric={metric}...")

                # Perform DBSCAN clustering
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
                labels = dbscan.fit_predict(embeddings_scaled)

                # Only calculate silhouette score if valid clusters are found
                if len(set(labels)) > 1:
                    silhouette_avg = silhouette_score(embeddings_scaled, labels)
                    silhouette_values = silhouette_samples(embeddings_scaled, labels)
                else:
                    silhouette_avg = -1  # Invalid configuration, no clusters

                # Update the best configuration if this one is better
                if silhouette_avg > best_silhouette_score:
                    best_silhouette_score = silhouette_avg
                    best_eps = eps
                    best_min_samples = min_samples
                    best_metric = metric
                    best_umap_file = umap_file
                    best_labels = labels
                    best_data = data.copy()  # Save a copy of the best data

                    # Append cluster labels and silhouette scores to the best data
                    for i, record in enumerate(best_data):
                        record['cluster_label'] = int(labels[i])
                        record['silhouette_coefficient'] = float(silhouette_values[i])

print("Best configuration identified!")
print(f"Best UMAP file: {best_umap_file}")
print(f"Best DBSCAN parameters: eps={best_eps}, min_samples={best_min_samples}, metric={best_metric}")
print(f"Best Silhouette Score: {best_silhouette_score:.4f}")

# Compute t-SNE embeddings for the best configuration
print("Computing t-SNE embeddings for the best configuration...")
tsne = TSNE(n_components=2, random_state=42)
best_tsne_embeddings = tsne.fit_transform(embeddings_scaled)

# Plot t-SNE with the best DBSCAN clusters
print("Generating t-SNE visualization...")
plt.figure(figsize=(10, 8))
plt.scatter(best_tsne_embeddings[:, 0], best_tsne_embeddings[:, 1], c=best_labels, cmap='Spectral', s=10)
plt.colorbar(label='DBSCAN Cluster Label')
plt.title(f'Best DBSCAN Clustering (t-SNE Visualization) - eps={best_eps}, min_samples={best_min_samples}, metric={best_metric}')
tsne_output_file = os.path.join(output_dir, f'best_t-sne_DBSCAN_eps_{best_eps}_min_samples_{best_min_samples}_metric_{best_metric}.png')
plt.savefig(tsne_output_file, dpi=300)
plt.close()
print(f"t-SNE visualization saved to {tsne_output_file}")

# Save the clustered data 
print("Saving the best clustered data...")
best_output_file = os.path.join(output_dir, f'best_clusteredData_DBSCAN_eps_{best_eps}_min_samples_{best_min_samples}_metric_{best_metric}.json')
with open(best_output_file, 'w') as f:
    json.dump({
        "mean_silhouette_score": best_silhouette_score,
        "records": best_data
    }, f, indent=4)
print(f"Best clustered data saved to {best_output_file}")
