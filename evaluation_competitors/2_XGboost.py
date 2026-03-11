import logging
import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

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
log_filename = f"3_xgboost_gemma_{timestamp}.log"
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
    try:
        logger.info("Loading train, validation and test datasets...")

        df_train = pd.read_parquet(os.path.join(OUTPUT_DIR, 'NEW_OSHA_DATABASE_filtered_embeddings_train.parquet'))
        df_validation = pd.read_parquet(os.path.join(OUTPUT_DIR, 'NEW_OSHA_DATABASE_filtered_embeddings_val.parquet'))
        df_test = pd.read_parquet(os.path.join(OUTPUT_DIR, 'NEW_OSHA_DATABASE_filtered_embeddings_test.parquet'))

        logger.info(f"Loaded datasets - Train: {df_train.shape}, Val: {df_validation.shape}, Test: {df_test.shape}")
        logger.info(f"Using feature column: {X_COL}")
        logger.info(f"Class distribution in train: {df_train[Y_COL].value_counts().sort_index().to_dict()}")

        return df_train, df_validation, df_test

    except Exception as e:
        logger.exception(f"Error loading parquet files: {e}")
        raise


def prepare_features(df, X_COL, Y_COL):
    X = np.array([emb for emb in df[X_COL]])
    y = df[Y_COL].values
    logger.info(f"Features shape ({X_COL}): {X.shape}, Target shape: {y.shape}")
    return X, y

def create_xgboost_pipeline():
    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('xgb', xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            max_depth=2,
            learning_rate=1,
            random_state=RANDOM_SEED,
            eval_metric='mlogloss',
            verbosity=0
        ))
    ])

def train_xgboost_pipeline(pipeline, X_train, y_train, X_val, y_val):
    logger.info("Training XGBoost Pipeline...")

    pipeline.fit(X_train, y_train)

    val_pred = pipeline.predict(X_val)
    logger.info("Validation set classification report:")
    logger.info(classification_report(y_val, val_pred))

    return pipeline


def save_pipeline(pipeline, X_COL, pipeline_name):
    model_path = os.path.join(MODEL_DIR, f"{pipeline_name}.pkl")

    pipeline_data = {
        'pipeline': pipeline,
        'feature_col': X_COL,
        'metadata': {
            'trained_at': datetime.now().isoformat(),
            'random_seed': RANDOM_SEED,
            'n_features': pipeline.named_steps['xgb'].n_features_in_,
            'scaler_type': 'MinMaxScaler',
            'xgb_params': pipeline.named_steps['xgb'].get_params()
        }
    }

    with open(model_path, 'wb') as f:
        pickle.dump(pipeline_data, f)

    logger.info(f"XGBoost Pipeline saved to: {model_path}")
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
    logger.info("Evaluating XGBoost pipeline on test set...")

    y_pred = pipeline.predict(X_test)

    logger.info("\n" + "=" * 60)
    logger.info("TEST SET CLASSIFICATION REPORT - XGBOOST + GEMMA EMBEDDINGS")
    logger.info("=" * 60)
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    logger.info("\nConfusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))

    return y_pred


def main():
    logger.info("===== STARTING PIPELINE - XGBOOST + GEMMA EMBEDDINGS =====")

    X_COL = 'embeddings_berttiny'
    Y_COL = 'severity_level'

    df_train, df_validation, df_test = load_parquet_datasets(X_COL, Y_COL)

    X_train, y_train = prepare_features(df_train, X_COL, Y_COL)
    X_val, y_val = prepare_features(df_validation, X_COL, Y_COL)
    X_test, y_test = prepare_features(df_test, X_COL, Y_COL)

    pipeline = create_xgboost_pipeline()
    trained_pipeline = train_xgboost_pipeline(pipeline, X_train, y_train, X_val, y_val)

    model_path = save_pipeline(trained_pipeline, X_COL, "xgboost_gemma_pipeline")

    logger.info("Reloading XGBoost pipeline from filesystem...")
    loaded_pipeline = load_pipeline(model_path)

    evaluate_test_set(loaded_pipeline, X_test, y_test)

    logger.info("===== STEP 3 COMPLETE =====")


if __name__ == '__main__':
    main()
