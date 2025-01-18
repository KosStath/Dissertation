import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_distances


output_plot_path = r'C:\Users\kstat\Documents\Dissertation\Data\Methods\Method_2\Clustering\K-Means\K-Means_normalized\outlier_visualization.png'
output_outliers_path = r'C:\Users\kstat\Documents\Dissertation\Data\Methods\Method_2\Clustering\K-Means\K-Means_normalized\outliers.json'


with open(r'C:\Users\kstat\Documents\Dissertation\Data\Methods\Method_2\Clustering\K-Means\K-Means_normalized\clustered_embeddings_umap_embeddings_n10_md0.1_cosine_d50_6clusters.json', 'r') as f:
    clustered_data = json.load(f)


with open(r'C:\Users\kstat\Documents\Dissertation\Data\Methods\Method_2\UMAP\umap_embeddings_n10_md0.1_cosine_d50.json', 'r') as f:
    original_embeddings = json.load(f)

# Extract the 50D embeddings and map them to their ids
embeddings_dict = {entry['id']: entry['umap'] for entry in original_embeddings}

# Extract the records from clustered data
records = clustered_data['records']

# Step 1: Identify points with low silhouette coefficients
low_silhouette_threshold = 0.2  
outliers_silhouette = []

for record in records:
    if record['silhouette_coefficient'] < low_silhouette_threshold:
        outliers_silhouette.append(record['id'])

# Step 2: Calculate centroids for each cluster and identify points far from centroids
print("Calculating cluster centroids...")

# Initialize clusters with empty lists
cluster_centroids = {}

# Group points by cluster and initialize empty lists for each cluster
for record in records:
    cluster_label = record['cluster_label']
    # Initialize the list for the cluster if not already initialized
    if cluster_label not in cluster_centroids:
        cluster_centroids[cluster_label] = []

    # Append the point 
    cluster_centroids[cluster_label].append(embeddings_dict[record['id']])

# Calculate centroids
for cluster_label, points in cluster_centroids.items():
    cluster_centroids[cluster_label] = np.mean(points, axis=0)

print("Cluster centroids calculation completed.")

# Step 3: Calculate distances from centroid for each point
print("Identifying outliers based on centroid distance...")
outliers_centroid_distance = []
distance_threshold = 2.0  

for record in records:
    point = embeddings_dict[record['id']]
    cluster_label = record['cluster_label']
    centroid = cluster_centroids[cluster_label]
    
    # Cosine distance calculation
    distance = cosine_distances([point], [centroid])[0][0]
    
    if distance > distance_threshold:
        outliers_centroid_distance.append(record['id'])

# Combine silhouette-based and centroid-based outliers
outliers_combined = set(outliers_silhouette + outliers_centroid_distance)

# Step 4: Prepare UMAP data for visualization (use the 50D UMAP from the clustered results)
print("Preparing UMAP data for plotting...")
umap_data = np.array([record['umap'][:2] for record in records])  # Taking the first 2 dimensions for plotting
cluster_labels = np.array([record['cluster_label'] for record in records])

# Step 5: Plotting the results
print("Plotting outlier visualization...")
plt.figure(figsize=(10, 8))


plt.scatter(umap_data[:, 0], umap_data[:, 1], c=cluster_labels, cmap='viridis', s=10, alpha=0.8, label='Non-Outliers')


outlier_indices = [i for i, record in enumerate(records) if record['id'] in outliers_combined]
plt.scatter(umap_data[outlier_indices, 0], umap_data[outlier_indices, 1], c='red', s=20, alpha=0.9, label='Outliers')


plt.title("Outliers in Clustered Data (Highlighted in Red)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend()
plt.tight_layout()


plt.savefig(output_plot_path)
plt.close()

# Step 6: Save outlier data 
outliers_data = {
    "outliers_silhouette": outliers_silhouette,
    "outliers_centroid_distance": outliers_centroid_distance,
    "combined_outliers": list(outliers_combined)
}

with open(output_outliers_path, 'w') as f:
    json.dump(outliers_data, f)

print(f"Outlier visualization saved to: {output_plot_path}")
print(f"Outlier data saved to: {output_outliers_path}")
