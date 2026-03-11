import argparse
import logging
import os
import warnings  # Import warnings module
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans as pyclust_kmeans
from pyclustering.utils import distance_metric, type_metric
from scipy.stats import entropy
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split

# --- Patch numpy warnings to fix AttributeError ---
np.warnings = warnings

# -------------------------------
# CONSTANTS
# -------------------------------
RANDOM_SEED = 96
LOG_DIR = "log"

# -------------------------------
# LOGGING CONFIGURATION
# -------------------------------
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"3_clustering_single_run_{timestamp}.log"
log_filepath = os.path.join(LOG_DIR, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filepath, mode="w"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# -------------------------------
# Clustering Functions
# -------------------------------
def cosine_distance(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def kmeans_clustering(data, n_clusters, measure="cosine"):
    logger.info(f"Starting KMeans clustering with {n_clusters} clusters, using {measure} distance...")
    if measure == "cosine":
        metric = distance_metric(type_metric.USER_DEFINED, func=cosine_distance)
    elif measure == "euclidean":
        metric = distance_metric(type_metric.EUCLIDEAN)
    else:
        raise ValueError("Invalid distance measure")

    data_list = data.tolist()
    initial_centers = kmeans_plusplus_initializer(data_list, n_clusters).initialize()
    kmeans_instance = pyclust_kmeans(data_list, initial_centers, metric=metric)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    centers = kmeans_instance.get_centers()

    labels = np.zeros(len(data_list), dtype=int)
    for cluster_idx, cluster_points in enumerate(clusters):
        for point_idx in cluster_points:
            labels[point_idx] = cluster_idx

    silhouette = silhouette_score(data, labels) if len(set(labels)) > 1 else -1
    davies_bouldin = davies_bouldin_score(data, labels) if len(set(labels)) > 1 else np.inf
    calinski_harabasz = calinski_harabasz_score(data, labels) if len(set(labels)) > 1 else -1

    logger.info(f"KMeans clustering results (k={n_clusters}): Silhouette={silhouette:.4f}, "
                f"Davies-Bouldin={davies_bouldin:.4f}, Calinski-Harabasz={calinski_harabasz:.4f}")

    return kmeans_instance, centers, labels, {
        "silhouette_score": silhouette,
        "davies_bouldin_score": davies_bouldin,
        "calinski_harabasz_score": calinski_harabasz,
    }


def compute_cluster_entropies(df, cluster_col='cluster', label_col='severity_level'):
    cluster_entropy = {}
    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster_id]
        label_counts = cluster_data[label_col].value_counts(normalize=True)
        cluster_entropy[cluster_id] = entropy(label_counts)
        logger.info(f"Entropy for cluster {cluster_id}: {cluster_entropy[cluster_id]:.4f}")

    mean_entropy = np.mean(list(cluster_entropy.values()))
    num_labels = len(df[label_col].unique())
    max_entropy = np.log(num_labels) if num_labels > 0 else 1
    normalized_entropy = {k: v / max_entropy for k, v in cluster_entropy.items()}
    mean_normalized_entropy = np.mean(list(normalized_entropy.values()))

    logger.info(f"Mean entropy across clusters: {mean_entropy:.4f}")
    logger.info(f"Mean normalized entropy across clusters: {mean_normalized_entropy:.4f}")

    return cluster_entropy, mean_entropy, normalized_entropy, mean_normalized_entropy


# -------------------------------
# Single Train/Test Split Function
# -------------------------------
def single_split_clustering(df,
                            embedding_col,
                            label_col,
                            n_clusters,
                            output_dir='output',
                            save_models=True):
    logger.info("Splitting dataset into train/validation/test (60/20/20)...")
    df_temp, df_test = train_test_split(
        df,
        train_size=0.8,
        random_state=RANDOM_SEED,
        stratify=df[label_col]
    )

    df_train, df_validation = train_test_split(
        df_temp,
        train_size=0.8,
        random_state=RANDOM_SEED,
        stratify=df_temp[label_col]
    )

    logger.info(f"Train set: {len(df_train)} samples ({len(df_train) / len(df) * 100:.1f}%)")
    logger.info(f"Validation set: {len(df_validation)} samples ({len(df_validation) / len(df) * 100:.1f}%)")
    logger.info(f"Test set: {len(df_test)} samples ({len(df_test) / len(df) * 100:.1f}%)")

    # Extract features
    X_train = pd.DataFrame(df_train[embedding_col].tolist()).values
    X_val = pd.DataFrame(df_validation[embedding_col].tolist()).values
    X_test = pd.DataFrame(df_test[embedding_col].tolist()).values

    y_train = df_train[label_col].values
    y_val = df_validation[label_col].values
    y_test = df_test[label_col].values

    logger.info(f"Using embedding column '{embedding_col}' with shapes:")
    logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Train clustering on train set
    kmeans, centers, train_labels, train_scores = kmeans_clustering(X_train, n_clusters)

    # Save model and centers
    if save_models:
        kmeans_path = os.path.join(output_dir, 'cluster_kmeans.joblib')
        centers_path = os.path.join(output_dir, 'cluster_centers.npy')
        joblib.dump(kmeans, kmeans_path)
        np.save(centers_path, centers)
        logger.info(f"KMeans model saved to {kmeans_path}")
        logger.info(f"Centers saved to {centers_path}")

    # Predict clusters on validation and test sets
    def predict_clusters(X, centers):
        labels = []
        for x in X:
            distances = [cosine_distance(x, center) for center in centers]
            labels.append(np.argmin(distances))
        return np.array(labels)

    labels_train = predict_clusters(X_train, centers)
    labels_val = predict_clusters(X_val, centers)
    labels_test = predict_clusters(X_test, centers)

    # Create full datasets with predicted clusters
    df_train_full = df_train.copy()
    df_train_full['cluster'] = labels_train

    df_val_full = df_validation.copy()
    df_val_full['cluster'] = labels_val

    df_test_full = df_test.copy()
    df_test_full['cluster'] = labels_test

    # Save datasets
    train_parquet_path = os.path.join(output_dir, 'cluster_train.parquet')
    val_parquet_path = os.path.join(output_dir, 'cluster_validation.parquet')
    test_parquet_path = os.path.join(output_dir, 'cluster_test.parquet')

    df_train_full.to_parquet(train_parquet_path, engine='pyarrow')
    df_val_full.to_parquet(val_parquet_path, engine='pyarrow')
    df_test_full.to_parquet(test_parquet_path, engine='pyarrow')

    logger.info(f"Validation data with predicted clusters saved to {val_parquet_path}")
    logger.info(f"Test data with predicted clusters saved to {test_parquet_path}")

    # Calculate metrics for validation and test sets
    def compute_metrics(X, labels):
        if len(set(labels)) > 1:
            silhouette = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
        else:
            silhouette, davies_bouldin, calinski_harabasz = -1, np.inf, -1
        return silhouette, davies_bouldin, calinski_harabasz

    val_silhouette, val_davies, val_calinski = compute_metrics(X_val, labels_val)
    test_silhouette, test_davies, test_calinski = compute_metrics(X_test, labels_test)

    logger.info(f"Validation scores: Silhouette: {val_silhouette:.4f}, "
                f"Davies-Bouldin: {val_davies:.4f}, Calinski-Harabasz: {val_calinski:.4f}")
    logger.info(f"Test scores: Silhouette: {test_silhouette:.4f}, "
                f"Davies-Bouldin: {test_davies:.4f}, Calinski-Harabasz: {test_calinski:.4f}")

    # Compute entropy metrics for validation and test
    df_val_metrics = pd.DataFrame({'cluster': labels_val, label_col: y_val})
    df_test_metrics = pd.DataFrame({'cluster': labels_test, label_col: y_test})

    val_entropies = compute_cluster_entropies(df_val_metrics, cluster_col='cluster', label_col=label_col)
    test_entropies = compute_cluster_entropies(df_test_metrics, cluster_col='cluster', label_col=label_col)

    results = {
        # Train scores
        "train_scores": train_scores,
        # Validation scores
        "val_silhouette_score": val_silhouette,
        "val_davies_bouldin_score": val_davies,
        "val_calinski_harabasz_score": val_calinski,
        "val_mean_normalized_entropy": val_entropies[3],
        # Test scores
        "test_silhouette_score": test_silhouette,
        "test_davies_bouldin_score": test_davies,
        "test_calinski_harabasz_score": test_calinski,
        "test_mean_normalized_entropy": test_entropies[3],
        # Paths
        "kmeans_path": kmeans_path if save_models else None,
        "centers_path": centers_path if save_models else None,
        "val_parquet_path": val_parquet_path,
        "test_parquet_path": test_parquet_path,
    }

    return results


# -------------------------------
# MAIN
# -------------------------------
def main(num_clusters):
    logger.info("===== STARTING PIPELINE - STEP 3 CLUSTERING (SINGLE SPLIT) =====")

    logger.info("\nStage 1: Loading dataset with reduced embeddings...")
    df_osha = pd.read_parquet('output/NEW_OSHA_DATABASE_FILTERED_SIC_NAICS_embeddings_reduced.parquet')
    logger.info(f"Dataset loaded with {len(df_osha)} samples.")

    logger.info(f"\nStage 2: Running single train/test split clustering with {num_clusters} clusters...")
    results = single_split_clustering(df_osha,
                                      embedding_col='embeddings_reduced',
                                      label_col='severity_level',
                                      n_clusters=num_clusters)

    logger.info(f"Results for {num_clusters} clusters: {results}")

    logger.info("===== STEP 3 COMPLETE =====")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run clustering pipeline with specified number of clusters (single split).")
    parser.add_argument('--num_clusters',
                        type=int,
                        default=4,
                        help='Number of clusters')

    args = parser.parse_args()
    main(args.num_clusters)
