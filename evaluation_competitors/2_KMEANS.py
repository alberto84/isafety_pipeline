import logging
import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------- SEED TOTALE REPRODUCIBILITY --------
RANDOM_SEED = 96
np.random.seed(RANDOM_SEED)

# -------------------------------
# CONSTANTS
# -------------------------------
LOG_DIR = "log"
OUTPUT_DIR = "output"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
K_CLUSTERS = 3

# -------------------------------
# LOGGING CONFIGURATION
# -------------------------------
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"5_kmeans_clustering_{timestamp}.log"
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


def load_parquet_datasets(X_COL, Y_COL):
    df_train = pd.read_parquet(os.path.join(OUTPUT_DIR, 'NEW_OSHA_DATABASE_filtered_embeddings_train.parquet'))
    df_validation = pd.read_parquet(os.path.join(OUTPUT_DIR, 'NEW_OSHA_DATABASE_filtered_embeddings_val.parquet'))
    df_test = pd.read_parquet(os.path.join(OUTPUT_DIR, 'NEW_OSHA_DATABASE_filtered_embeddings_test.parquet'))

    logger.info(f"Loaded datasets - Train: {df_train.shape}, Val: {df_validation.shape}, Test: {df_test.shape}")
    logger.info(f"Train class distribution: {df_train[Y_COL].value_counts().sort_index().to_dict()}")

    return df_train, df_validation, df_test


def prepare_features(df, X_COL, Y_COL):
    """Estrae le features per clustering."""
    X = np.array([emb for emb in df[X_COL]])
    y_true = df[Y_COL].values
    logger.info(f"Features shape ({X_COL}): {X.shape}")
    return X, y_true


def create_kmeans_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=K_CLUSTERS, random_state=RANDOM_SEED, n_init=10))
    ])
    return pipeline


def fit_kmeans_pipeline(pipeline, X_train):
    logger.info(f"Fitting K-Means (k={K_CLUSTERS})...")
    pipeline.fit(X_train)
    logger.info("K-Means fitted successfully!")
    return pipeline


def evaluate_clustering(pipeline, X, y_true, dataset_name):
    cluster_labels = pipeline.named_steps['kmeans'].labels_

    ari = adjusted_rand_score(y_true, cluster_labels)
    silhouette = silhouette_score(X, cluster_labels)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"CLUSTERING RESULTS - {dataset_name}")
    logger.info(f"{'=' * 60}")
    logger.info(f"ARI Score: {ari:.4f}")
    logger.info(f"Silhouette Score: {silhouette:.4f}")
    logger.info(f"Cluster sizes: {np.bincount(cluster_labels)}")
    logger.info(f"True class distribution: {np.bincount(y_true)}")

    cm = confusion_matrix(y_true, cluster_labels)
    logger.info(f"\nConfusion Matrix (True vs Cluster):\n{cm}")

    return cluster_labels, ari, silhouette


def save_kmeans_pipeline(pipeline, feature_col, pipeline_name):
    model_path = os.path.join(MODEL_DIR, f"{pipeline_name}.pkl")

    pipeline_data = {
        'pipeline': pipeline,
        'feature_col': feature_col,
        'cluster_centers': pipeline.named_steps['kmeans'].cluster_centers_,
        'metadata': {
            'trained_at': datetime.now().isoformat(),
            'random_seed': RANDOM_SEED,
            'n_clusters': K_CLUSTERS,
            'n_features': pipeline.named_steps['kmeans'].n_features_in_,
            'inertia': pipeline.named_steps['kmeans'].inertia_
        }
    }

    with open(model_path, 'wb') as f:
        pickle.dump(pipeline_data, f)

    logger.info(f"K-Means pipeline saved to: {model_path}")
    return model_path


def load_kmeans_pipeline(model_path):
    with open(model_path, 'rb') as f:
        pipeline_data = pickle.load(f)

    logger.info(f"K-Means pipeline loaded from: {model_path}")
    logger.info(f"Metadata: {pipeline_data['metadata']}")
    return pipeline_data['pipeline']


def compute_cluster_statistics(df, y_pred_cluster, Y_COL, dataset_name="Dataset"):
    logger.info("\n" + "=" * 80)
    logger.info(f"DETAILED CLUSTER STATISTICS - {dataset_name}")
    logger.info("=" * 80)

    class_totals = df[Y_COL].value_counts().sort_index()
    total_records = len(df)

    logger.info(f"\nTotal records in {dataset_name}: {total_records}")
    logger.info(f"Class distribution in {dataset_name}:\n{class_totals}\n")

    all_stats = []

    for cluster_id in sorted(np.unique(y_pred_cluster)):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"CLUSTER {cluster_id}")
        logger.info(f"{'=' * 70}")

        cluster_mask = y_pred_cluster == cluster_id
        cluster_df = df[cluster_mask]
        cluster_size = len(cluster_df)

        logger.info(f"Total records in cluster: {cluster_size}")
        logger.info(f"% of total dataset: {100 * cluster_size / total_records:.2f}%\n")

        class_counts = cluster_df[Y_COL].value_counts().sort_index()

        logger.info(f"{'Class':<10} {'Count':<10} {'% in Cluster':<15} {'% of Class Total':<20}")
        logger.info("-" * 70)

        for class_id in sorted(df[Y_COL].unique()):
            count = class_counts.get(class_id, 0)
            pct_in_cluster = 100 * count / cluster_size if cluster_size > 0 else 0
            pct_of_class_total = 100 * count / class_totals[class_id]

            logger.info(f"{class_id:<10} {count:<10} {pct_in_cluster:<15.2f} {pct_of_class_total:<20.2f}")

            all_stats.append({
                'Cluster': cluster_id,
                'Class': class_id,
                'Count': count,
                '% in Cluster': round(pct_in_cluster, 2),
                '% of Class Total': round(pct_of_class_total, 2)
            })

    stats_df = pd.DataFrame(all_stats)

    return stats_df


def save_predictions_to_excel(df, y_pred_cluster, output_filename, dataset_name):
    df_output = df.copy()

    df_output['y_pred_cluster'] = y_pred_cluster

    cols_to_drop = [col for col in df_output.columns if 'embedding' in col.lower()]
    df_output_clean = df_output.drop(columns=cols_to_drop)

    excel_path = os.path.join(OUTPUT_DIR, output_filename)

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_output_clean.to_excel(writer, sheet_name=f'{dataset_name} Set', index=False)

    logger.info(f"\n{dataset_name} set with cluster predictions saved to: {excel_path}")
    logger.info(f"Sheet 1 '{dataset_name} Set': {len(df_output_clean)} rows, {len(df_output_clean.columns)} columns")

    return excel_path


def main():

    logger.info("===== STARTING PIPELINE - STEP 5 K-MEANS CLUSTERING (k=3) =====")

    X_COL = 'embeddings_tfidf'
    Y_COL = 'severity_level'

    df_train, df_validation, df_test = load_parquet_datasets(X_COL, Y_COL)

    X_train, y_train_true = prepare_features(df_train, X_COL, Y_COL)
    X_val, y_val_true = prepare_features(df_validation, X_COL, Y_COL)
    X_test, y_test_true = prepare_features(df_test, X_COL, Y_COL)

    pipeline = create_kmeans_pipeline()
    trained_pipeline = fit_kmeans_pipeline(pipeline, X_train)

    train_clusters = trained_pipeline.named_steps['kmeans'].labels_
    train_ari = adjusted_rand_score(y_train_true, train_clusters)
    logger.info(f"Train ARI score: {train_ari:.4f}")

    compute_cluster_statistics(df_train, train_clusters, Y_COL, "TRAIN")

    save_predictions_to_excel(
        df_train,
        train_clusters,
        f"train_set_kmeans_predictions_{timestamp}.xlsx",
        "Train"
    )

    model_path = save_kmeans_pipeline(trained_pipeline, X_COL, "kmeans_tfidf_k3")

    logger.info("Reloading K-Means pipeline from filesystem...")
    loaded_pipeline = load_kmeans_pipeline(model_path)

    test_clusters_reloaded = loaded_pipeline.predict(X_test)
    reloaded_ari = adjusted_rand_score(y_test_true, test_clusters_reloaded)
    logger.info(f"Reloaded pipeline ARI on test: {reloaded_ari:.4f} ✓")

    save_predictions_to_excel(
        df_test,
        test_clusters_reloaded,
        f"test_set_kmeans_predictions_{timestamp}.xlsx",
        "Test"
    )

    df_test['y_pred_cluster'] = test_clusters_reloaded
    mapping = {0: 2,
               1: 0,
               2: 1}
    df_test['cluster_assigned_class'] = df_test['y_pred_cluster'].apply(lambda x: mapping[x])
    y_pred = df_test['cluster_assigned_class']
    print(classification_report(y_test_true, y_pred, digits=4, zero_division=0))
    logger.info("===== STEP 5 COMPLETE =====")


if __name__ == '__main__':
    main()
