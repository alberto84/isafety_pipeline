import logging
import os
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# CONSTANTS
# -------------------------------
RANDOM_SEED = 96
LOG_DIR = "log"
OUTPUT_DIR = "output/models"

# -------------------------------
# LOGGING CONFIGURATION
# -------------------------------
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"2_classifier_{timestamp}.log"
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


def split_train_validation_test(df_osha):
    """Split train-validation-test 60/20/20 using exact user-specified method"""
    logger.info("Splitting dataset...")

    # Split train-validation-test 60/20/20
    df_temp, df_test = train_test_split(
        df_osha,
        train_size=0.8,
        random_state=RANDOM_SEED,
        stratify=df_osha['severity_level']
    )

    df_train, df_validation = train_test_split(
        df_temp,
        train_size=0.8,
        random_state=RANDOM_SEED,
        stratify=df_temp['severity_level']
    )

    return df_train, df_validation, df_test


def extract_features_target(df_split, split_name, X_COL, Y_COL):
    """Extract embeddings (X) and severity_level (y) from dataframe split"""
    X = np.vstack(df_split[X_COL].values)
    y = df_split[Y_COL].values

    logger.info(f"{split_name} - X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"{split_name} severity dist:\n{pd.Series(y).value_counts().sort_index()}")

    return X, y


# -------------------------------
# MODEL DEFINITIONS & TRAINING
# -------------------------------
def train_random_forest(X_train, X_val, y_train, y_val):
    """Train Random Forest model"""
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)

    val_score = rf.score(X_val, y_val)
    logger.info(f"RF Validation accuracy: {val_score:.4f}")
    return rf, 'random_forest'


def train_decision_tree(X_train, X_val, y_train, y_val):
    """Train Decision Tree model"""
    logger.info("Training Decision Tree...")
    dt = DecisionTreeClassifier(random_state=RANDOM_SEED)
    dt.fit(X_train, y_train)

    val_score = dt.score(X_val, y_val)
    logger.info(f"DT Validation accuracy: {val_score:.4f}")
    return dt, 'decision_tree'


def train_xgboost(X_train, X_val, y_train, y_val):
    """Train XGBoost model"""
    logger.info("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=2,
        max_depth=2,
        learning_rate=1,
        random_state=RANDOM_SEED,
        eval_metric='mlogloss'
    )
    xgb.fit(X_train, y_train)

    val_score = xgb.score(X_val, y_val)
    logger.info(f"XGB Validation accuracy: {val_score:.4f}")
    return xgb, 'xgboost'


def train_knn(X_train, X_val, y_train, y_val):
    """Train KNN model"""
    logger.info("Training KNN...")
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)

    val_score = knn.score(X_val, y_val)
    logger.info(f"KNN Validation accuracy: {val_score:.4f}")
    return knn, 'knn'


# -------------------------------
# EVALUATION & VISUALIZATION
# -------------------------------
def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model on test set and return metrics"""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0, output_dict=True)
    report2 = classification_report(y_test, y_pred, digits=4, zero_division=0)

    logger.info(f"\n{model_name.upper()} Test Results:")
    logger.info(report2)

    return {
        'model': model_name,
        'accuracy': report['accuracy'],
        'f1_weighted': report['weighted avg']['f1-score'],
        'classification_report': report,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }


def plot_confusion_matrix(cm, model_name, output_dir):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()


# -------------------------------
# MAIN PIPELINE - 2 Classifier
# -------------------------------
def main():
    logger.info("===== STARTING PIPELINE - STEP 2 CLASSIFIER =====")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Stage 1: Load data and split 60/20/20
    logger.info("\nStage 1: Loading data and splitting 60/20/20...")
    df_osha = pd.read_parquet('output/NEW_OSHA_DATABASE_FILTERED_SIC_NAICS_embeddings.parquet')
    df_train, df_validation, df_test = split_train_validation_test(df_osha)

    # Extract features for all splits
    X_COL = 'embeddings'
    Y_COL = 'severity_level'
    X_train, y_train = extract_features_target(df_train, "Train", X_COL, Y_COL)
    X_val, y_val = extract_features_target(df_validation, "Validation", X_COL, Y_COL)
    X_test, y_test = extract_features_target(df_test, "Test", X_COL, Y_COL)

    # Stage 2: Train all models
    logger.info("\nStage 2: Training models...")
    models = {}
    models['random_forest'] = train_random_forest(X_train, X_val, y_train, y_val)
    models['decision_tree'] = train_decision_tree(X_train, X_val, y_train, y_val)
    models['xgboost'] = train_xgboost(X_train, X_val, y_train, y_val)
    models['knn'] = train_knn(X_train, X_val, y_train, y_val)

    # Stage 3: Evaluate all models
    logger.info("\nStage 3: Evaluating models on test set...")
    results = {}
    for model_name, (model, _) in models.items():
        logger.info(f"=== Evaluating {model_name}... ===")
        results[model_name] = evaluate_model(model, X_test, y_test, model_name)

        # Plot confusion matrix
        plot_confusion_matrix(
            results[model_name]['confusion_matrix'],
            model_name,
            OUTPUT_DIR
        )

    logger.info("===== STEP 2 COMPLETE =====")


if __name__ == '__main__':
    main()
