import json
import ijson
import numpy as np
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm  
from sklearn.preprocessing import StandardScaler
import os

# Function to process the JSON file and extract embeddings in batches
def process_json(input_file_path, batch_size=1000):
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        objects = ijson.items(infile, 'item')  
        batch = []
        count = 0
        for obj in objects:
            count += 1
            abstract_id = obj["id"]
            year = obj["year"]
            embedding = obj["embedding"]
            # Only append if the embedding is valid (all-zero embeddings are valid too)
            batch.append({
                "id": abstract_id,
                "year": year,
                "embedding": embedding
            })

            # Yield the batch and reset when it reaches the batch size
            if count % batch_size == 0:
                yield batch
                batch = []
        # Yield the last batch if it's smaller than the batch size
        if batch:
            yield batch

# Function to apply UMAP with different parameter combinations
def apply_umap(embeddings, n_neighbors, min_dist, metric, n_components):
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                            n_components=n_components, random_state=42)
    umap_embeddings = umap_model.fit_transform(embeddings)
    return umap_embeddings


def save_results(output_file_path, results):
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=4, ensure_ascii=False)

# Function to create and save the visualizations
def save_visualizations(umap_embeddings, output_dir, n_neighbors, min_dist, metric, n_components):
    
    os.makedirs(output_dir, exist_ok=True)
    
    if n_components == 2:
        # Plot 2D
        plt.figure(figsize=(8, 6))
        plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], s=1)
        plt.title(f"UMAP 2D Embedding\nn_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.savefig(f"{output_dir}/umap_2d_n{n_neighbors}_md{min_dist}_{metric}.png", dpi=300)
        plt.close()
    elif n_components == 3:
        # Plot 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], umap_embeddings[:, 2], s=1)
        ax.set_title(f"UMAP 3D Embedding\nn_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")
        plt.savefig(f"{output_dir}/umap_3d_n{n_neighbors}_md{min_dist}_{metric}.png", dpi=300)
        plt.close()

# Main function to process, apply UMAP, and save results for specified parameter combinations
def process_and_generate_umap(input_file_path, output_dir, batch_size=1000):
    
    n_neighbors_list = [5, 10, 15, 30]
    min_dist_list = [0.1, 0.3, 0.5]
    metrics = "cosine"
    n_components_list = [2, 3, 5, 10]  
    
    embeddings_batch = []
    abstract_ids = []
    years = []

    # Process data in batches and accumulate embeddings
    print("Processing data and accumulating embeddings...")
    for batch in tqdm(process_json(input_file_path, batch_size)):
        for entry in batch:
            embedding = entry["embedding"]
            if embedding != [0] * len(embedding):  # Ignore all-zero embeddings for processing
                # Convert Decimal to float
                embedding = [float(value) for value in embedding]
                embeddings_batch.append(embedding)
                abstract_ids.append(entry["id"])
                years.append(entry["year"])

    # Convert embeddings_batch to a NumPy array
    embeddings_batch = np.array(embeddings_batch)

    # Add noise to the embeddings
    noise = np.random.normal(0, 0.01, embeddings_batch.shape)  
    embeddings_batch += noise

    # Standardize the data
    scaler = StandardScaler()
    embeddings_batch = scaler.fit_transform(embeddings_batch)

    # Apply UMAP for each parameter combination
    for n_neighbors in n_neighbors_list:
        for min_dist in min_dist_list:
            for n_components in n_components_list:  
                print(f"Applying UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metrics}, n_components={n_components}")
                umap_embeddings = apply_umap(embeddings_batch, n_neighbors, min_dist, metrics, n_components)

                # Collect results
                results = []
                for idx, abstract_id in enumerate(abstract_ids):
                    results.append({
                        "id": abstract_id,
                        "year": years[idx],
                        "umap_embedding": umap_embeddings[idx].tolist()  # Store the embedding
                    })

               
                result_file = os.path.join(output_dir, f"umap_embeddings_n{n_neighbors}_md{min_dist}_nc{n_components}_{metrics}.json")
                save_results(result_file, results)

                
                save_visualizations(umap_embeddings, output_dir, n_neighbors, min_dist, metrics, n_components)

    print(f"UMAP process completed. Results saved to {output_dir}.")


input_file_path = r'C:\Users\kstat\Documents\Dissertation\Data\Methods\Method_2\Clustering_SemiSupervised\weightedAbstractEmbed_2.json'
output_dir = r'C:\Users\kstat\Documents\Dissertation\Data\Methods\Method_2\Clustering_SemiSupervised\UMAP_2'


process_and_generate_umap(input_file_path, output_dir, batch_size=1000)