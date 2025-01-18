import os
import json
import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist


input_file = r"C:\Users\kstat\Documents\Dissertation\Data\Methods\Method_2\Clustering_SemiSupervised\ClusterAnalysis\clustered_embeddings_umap_embeddings_n30_md0.5_nc10_cosine_10clusters.json"
keywords_file = r"C:\Users\kstat\Documents\Dissertation\Data\datasetWithKeys.json"
output_folder = r"C:\Users\kstat\Documents\Dissertation\Data\Methods\Method_2\Clustering_SemiSupervised\ClusterAnalysis"

os.makedirs(output_folder, exist_ok=True)


with open(keywords_file, "r", encoding="utf-8") as f:
    keywords_data = json.load(f)
keywords_map = {record["id"]: record.get("Extracted_Keywords", "").split(", ") for record in keywords_data}

# Helper function to save JSON output
def save_output(output_data, filename):
    output_path = os.path.join(output_folder, filename)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    return output_path

# Analysis for the input file
with open(input_file, "r") as f:
    clustered_data = json.load(f)

records = clustered_data["records"]
mean_silhouette_score = clustered_data["mean_silhouette_score"]

# Initialize metrics
cluster_counts = Counter()
cluster_keywords = {}
cluster_silhouettes = {}
cluster_umap_embeddings = {}


for record in records:
    cluster_label = record["cluster_label"]
    cluster_counts[cluster_label] += 1

    # Link keywords by ID
    record_id = record["id"]
    keywords = keywords_map.get(record_id, [])
    cluster_keywords.setdefault(cluster_label, []).extend(keywords)

    # Silhouette coefficients
    silhouette_coeff = record["silhouette_coefficient"]
    cluster_silhouettes.setdefault(cluster_label, []).append(silhouette_coeff)

    # UMAP embeddings
    umap_embedding = np.array(record["umap_embedding"])
    cluster_umap_embeddings.setdefault(cluster_label, []).append(umap_embedding)

# Metrics Calculations
# 1. Record Counts
record_counts_output = dict(cluster_counts)

# 2. Keywords per Cluster
keyword_stats_output = {}
total_keywords_dataset = sum(len(kw) for kw in cluster_keywords.values())
unique_keywords_dataset = len(set(kw for kws in cluster_keywords.values() for kw in kws))

for cluster, keywords in cluster_keywords.items():
    total_keywords_cluster = len(keywords)
    unique_keywords_cluster = len(set(keywords))
    keyword_counts = Counter(keywords)
    top_keywords = keyword_counts.most_common(10)  # Top 10 keywords
    diversity = unique_keywords_cluster / total_keywords_cluster

    keyword_stats_output[cluster] = {
        "total_keywords": total_keywords_cluster,
        "unique_keywords": unique_keywords_cluster,
        "top_keywords": top_keywords,
        "diversity": diversity,
        "relative_diversity": diversity / (unique_keywords_dataset / total_keywords_dataset),
    }

# 3. Silhouette Coefficients
silhouette_stats_output = {}
for cluster, silhouettes in cluster_silhouettes.items():
    silhouettes = np.array(silhouettes)
    silhouette_stats_output[cluster] = {
        "mean": np.mean(silhouettes),
        "median": np.median(silhouettes),
        "min": np.min(silhouettes),
        "max": np.max(silhouettes),
        "std_dev": np.std(silhouettes),
    }

# 4. Cluster Centroids
centroids_output = {}
for cluster, embeddings in cluster_umap_embeddings.items():
    embeddings = np.array(embeddings)
    centroid = np.mean(embeddings, axis=0).tolist()
    centroids_output[cluster] = centroid

# 5. Cluster Homogeneity
homogeneity_output = {}
for cluster, embeddings in cluster_umap_embeddings.items():
    embeddings = np.array(embeddings)
    centroid = np.array(centroids_output[cluster])
    distances = cdist(embeddings, centroid.reshape(1, -1))
    homogeneity_output[cluster] = {
        "mean_distance": np.mean(distances),
        "std_dev_distance": np.std(distances),
    }

# 6. Cluster Size Ratios
size_ratios_output = {cluster: count / sum(cluster_counts.values()) for cluster, count in cluster_counts.items()}


base_filename = os.path.basename(input_file).replace(".json", "")
save_output(record_counts_output, f"record_counts_{base_filename}.json")
save_output(keyword_stats_output, f"keyword_stats_{base_filename}.json")
save_output(silhouette_stats_output, f"silhouette_coefficients_{base_filename}.json")
save_output(centroids_output, f"cluster_centroids_{base_filename}.json")
save_output(homogeneity_output, f"cluster_homogeneity_{base_filename}.json")
save_output(size_ratios_output, f"cluster_size_ratios_{base_filename}.json")
