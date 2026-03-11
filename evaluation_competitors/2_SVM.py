import logging
import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# -------- SEED TOTALE REPRODUCIBILITY --------
RANDOM_SEED = 96
np.random.seed(RANDOM_SEED)

# -------------------------------
# CONSTANTS
# -------------------------------
LOG_DIR = "log"
OUTPUT_DIR = "output"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

# -------------------------------
# LOGGING CONFIGURATION
# -------------------------------
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"2_svm_pipeline_{timestamp}.log"
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


def load_parquet_datasets():
    try:
        logger.info("Loading train, validation and test datasets...")

        df_train = pd.read_parquet(os.path.join(OUTPUT_DIR, 'NEW_OSHA_DATABASE_filtered_embeddings_train.parquet'))
        df_validation = pd.read_parquet(os.path.join(OUTPUT_DIR, 'NEW_OSHA_DATABASE_filtered_embeddings_val.parquet'))
        df_test = pd.read_parquet(os.path.join(OUTPUT_DIR, 'NEW_OSHA_DATABASE_filtered_embeddings_test.parquet'))

        logger.info(f"Loaded datasets - Train: {df_train.shape}, Val: {df_validation.shape}, Test: {df_test.shape}")
        logger.info(f"Class distribution in train: {df_train['severity_level'].value_counts().sort_index().to_dict()}")

        return df_train, df_validation, df_test

    except Exception as e:
        logger.exception(f"Error loading parquet files: {e}")
        raise


def prepare_features(df, X_col='embeddings_tfidf', Y_col='severity_level'):
    X = np.array([emb for emb in df[X_col]])
    y = df[Y_col].values
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y


def create_svm_pipeline():
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('svm', SVC(gamma='auto', random_state=RANDOM_SEED))
    ])
    return pipeline


def train_svm_pipeline(pipeline, X_train, y_train, X_val, y_val):
    logger.info("Training SVM Pipeline...")

    pipeline.fit(X_train, y_train)

    val_pred = pipeline.predict(X_val)
    logger.info("Validation set classification report:")
    logger.info(classification_report(y_val, val_pred, digits=4, zero_division=0))

    return pipeline


def save_pipeline(pipeline, pipeline_name):
    model_path = os.path.join(MODEL_DIR, f"{pipeline_name}.pkl")

    pipeline_data = {
        'pipeline': pipeline,
        'feature_col': 'embeddings_tfidf',
        'metadata': {
            'trained_at': datetime.now().isoformat(),
            'random_seed': RANDOM_SEED,
            'n_features': pipeline.named_steps['svm'].n_features_in_ if hasattr(pipeline.named_steps['svm'],
                                                                                'n_features_in_') else
            pipeline[1].shape[1],
            'scaler_type': 'MinMaxScaler',
            'svm_params': pipeline.named_steps['svm'].get_params()
        }
    }

    with open(model_path, 'wb') as f:
        pickle.dump(pipeline_data, f)

    logger.info(f"SVM Pipeline saved to: {model_path}")
    return model_path


def load_pipeline(model_path):
    try:
        with open(model_path, 'rb') as f:
            pipeline_data = pickle.load(f)

        logger.info(f"Pipeline loaded from: {model_path}")
        logger.info(f"Pipeline metadata: {pipeline_data['metadata']}")

        return pipeline_data['pipeline']

    except Exception as e:
        logger.exception(f"Error loading pipeline: {e}")
        raise


def evaluate_test_set(pipeline, X_test, y_test):
    logger.info("Evaluating pipeline on test set...")

    y_pred = pipeline.predict(X_test)

    logger.info("\n" + "=" * 60)
    logger.info("TEST SET CLASSIFICATION REPORT")
    logger.info("=" * 60)
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    logger.info("\nConfusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))

    return y_pred


def main():
    logger.info("===== STARTING PIPELINE - SVM PIPELINE CLASSIFIER =====")

    X_COL = 'embeddings_tfidf'
    Y_COL = 'severity_level'

    df_train, df_validation, df_test = load_parquet_datasets()

    X_train, y_train = prepare_features(df_train, X_COL, Y_COL)
    X_val, y_val = prepare_features(df_validation, X_COL, Y_COL)
    X_test, y_test = prepare_features(df_test, X_COL, Y_COL)

    pipeline = create_svm_pipeline()
    trained_pipeline = train_svm_pipeline(pipeline, X_train, y_train, X_val, y_val)

    model_path = save_pipeline(trained_pipeline, "svm_pipeline_tfidf")

    logger.info("Reloading pipeline from filesystem...")
    loaded_pipeline = load_pipeline(model_path)

    evaluate_test_set(loaded_pipeline, X_test, y_test)

    logger.info("===== STEP COMPLETE =====")


if __name__ == '__main__':
    main()
