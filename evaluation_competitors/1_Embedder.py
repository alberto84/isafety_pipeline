import logging
import os
import random
import re
from datetime import datetime

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

# -------- SEED TOTALE REPRODUCIBILITY --------
RANDOM_SEED = 96
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -------------------------------
# CONSTANTS
# -------------------------------
LOG_DIR = "log"

SEVERITY_RANK = {
    "Fatality": 0,
    "Hospitalized injury": 1,
    "Non Hospitalized injury": 2,
    None: 3
}
MIN_SEVERITY_RANK = 3
OUTPUT_DIR = "output"

# -------------------------------
# LOGGING CONFIGURATION
# -------------------------------
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"1_embedder_{timestamp}.log"
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
# DATA LOADING
# -------------------------------
def load_excel(file_path: str) -> pd.ExcelFile:
    try:
        logger.info(f"Loading Excel file: {file_path}")
        excel_data = pd.ExcelFile(file_path)
        logger.info(f"Sheets found: {excel_data.sheet_names}")
        return excel_data
    except Exception as e:
        logger.exception(f"Error loading Excel file: {e}")
        raise


def extract_dataset(excel_data: pd.ExcelFile, sheet_index: int) -> pd.DataFrame:
    try:
        sheet_name = excel_data.sheet_names[sheet_index]
        logger.info(f"Extracting data from sheet {sheet_index + 1}: '{sheet_name}'")
        df = excel_data.parse(sheet_name=sheet_index)
        logger.info(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        logger.exception(f"Error extracting sheet {sheet_index + 1}: {e}")
        raise


# -------------------------------
# TEXT PREPROCESSING
# -------------------------------
def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def print_class_counts(df_train, df_validation, df_test):

    # Stampa conteggi per classe
    logger.info("Train set class counts:")
    logger.info(df_train['severity_level'].value_counts().sort_index())

    logger.info("\nValidation set class counts:")
    logger.info(df_validation['severity_level'].value_counts().sort_index())

    logger.info("\nTest set class counts:")
    logger.info(df_test['severity_level'].value_counts().sort_index())


# -------------------------------
# DATASET COMBINATION
# -------------------------------
def combine_dataset(df_data: pd.DataFrame, df_employee: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Preparing employee dataset for left join with 'max severity' logic...")

        required_cols = {"Degree", "report_num"}
        if not required_cols.issubset(df_employee.columns):
            raise KeyError(f"Colonne {required_cols} mancanti in df_employee")

        df_employee = df_employee.copy()
        df_employee["Degree"] = df_employee["Degree"].where(df_employee["Degree"].notna(), None)

        df_employee["severity_level"] = df_employee["Degree"].map(lambda x: SEVERITY_RANK.get(x, MIN_SEVERITY_RANK))

        df_employee_reduced = (df_employee
                               .sort_values("severity_level")
                               .drop_duplicates(subset=["report_num"], keep="first")
                               )

        logger.info(f"Reduced employee dataset to {df_employee_reduced.shape[0]} rows with most severe Degree.")

        df_merged = pd.merge(
            df_data,
            df_employee_reduced,
            on='report_num',
            how='left'
        )

        logger.info(f"Joined dataset contains {df_merged.shape[0]} rows and {df_merged.shape[1]} columns.")
        return df_merged

    except Exception as e:
        logger.error(f"Error during dataset combination: {e}")
        raise


# -------------------------------
# TF-IDF EMBEDDING
# -------------------------------
def fit_transform_tfidf(texts):
    """Fit TF-IDF vectorizer and transform texts"""
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95, ngram_range=(1, 1))
    tfidf_matrix = vectorizer.fit_transform(texts)
    embeddings = tfidf_matrix.toarray()
    return vectorizer, embeddings


def transform_tfidf(vectorizer, texts):
    """Transform texts to TF-IDF embeddings using fitted vectorizer"""
    tfidf_matrix = vectorizer.transform(texts)
    embeddings = tfidf_matrix.toarray()
    return embeddings


def select_k_best_chi2(X_train, y_train, X_val, X_test, k):
    est = KBinsDiscretizer(n_bins=30, encode='ordinal', strategy='uniform')
    est.fit(X_train)

    X_train_discretized = est.transform(X_train)
    X_val_discretized = est.transform(X_val)
    X_test_discretized = est.transform(X_test)

    chi2_selector = SelectKBest(chi2, k=k)
    chi2_selector.fit(X_train_discretized, y_train)
    selected_features_indices = chi2_selector.get_support(indices=True)

    X_new_train = X_train_discretized[:, selected_features_indices]
    X_new_val = X_val_discretized[:, selected_features_indices]
    X_new_test = X_test_discretized[:, selected_features_indices]

    return X_new_train, X_new_val, X_new_test, selected_features_indices


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main():

    logger.info("Stage 1: Loading datasets...")
    excel_osha = load_excel('../dataset/NEW_OSHA_DATABASE_FILTERED_SIC_NAICS.xlsx')
    df_data = extract_dataset(excel_osha, 0)
    df_employee = extract_dataset(excel_osha, 2)
    logger.info(f"OSHA dataset loaded with {df_data.shape[0]} rows and {df_data.shape[1]} columns.")

    df_osha = combine_dataset(df_data, df_employee)
    logger.info(f"Combined dataset with {df_osha.shape[0]} rows and {df_osha.shape[1]} columns.")

    # Preprocess the 'abstract' column
    logger.info("Preprocessing 'abstract' texts...")
    df_osha = df_osha.copy()
    df_osha['abstract'] = df_osha['abstract'].apply(preprocess_text)
    df_osha = df_osha[df_osha['severity_level'] != 3]

    logger.info("Splitting dataset...")
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

    logger.info(f"Dataset splitted in Train={len(df_train)}|Validation={len(df_validation)}|Test={len(df_test)}")
    print_class_counts(df_train, df_validation, df_test)

    # ========== TF-IDF + CHI2 ==========
    logger.info("Fitting TF-IDF vectorizer on training set...")
    vectorizer, train_embeddings = fit_transform_tfidf(df_train['abstract'].tolist())
    logger.info("Transforming validation and test set with trained TF-IDF vectorizer...")
    validation_embeddings = transform_tfidf(vectorizer, df_validation['abstract'].tolist())
    test_embeddings = transform_tfidf(vectorizer, df_test['abstract'].tolist())

    df_train = df_train.copy()
    df_validation = df_validation.copy()
    df_test = df_test.copy()

    df_train['embeddings_tfidf'] = list(train_embeddings)
    df_validation['embeddings_tfidf'] = list(validation_embeddings)
    df_test['embeddings_tfidf'] = list(test_embeddings)

    k = 128
    X_new_train, X_new_val, X_new_test, selected_features_indices = select_k_best_chi2(
        X_train=train_embeddings,
        y_train=df_train['severity_level'].values,
        X_val=validation_embeddings,
        X_test=test_embeddings,
        k=k
    )

    df_train['embeddings_reduced_tfidf'] = list(X_new_train)
    df_validation['embeddings_reduced_tfidf'] = list(X_new_val)
    df_test['embeddings_reduced_tfidf'] = list(X_new_test)

    logger.info(
        f"TF-IDF - Dimensione embeddings finale: train: {X_new_train.shape} | val: {X_new_val.shape} | test: {X_new_test.shape}")

    # ========== GEMMA EMBEDDINGS ==========
    logger.info("Loading GEMMA model and encoding abstracts...")
    model_gemma = SentenceTransformer('google/embeddinggemma-300m', use_auth_token='<HERE_YOUR_TOKEN>')
    emb_train_gemma = model_gemma.encode(df_train['abstract'].tolist())
    emb_val_gemma = model_gemma.encode(df_validation['abstract'].tolist())
    emb_test_gemma = model_gemma.encode(df_test['abstract'].tolist())

    df_train['embeddings_gemma'] = list(emb_train_gemma)
    df_validation['embeddings_gemma'] = list(emb_val_gemma)
    df_test['embeddings_gemma'] = list(emb_test_gemma)
    logger.info(f"GEMMA embeddings shape: train {emb_train_gemma.shape} | val {emb_val_gemma.shape} | test {emb_test_gemma.shape}")

    # ========== BERT-TINY EMBEDDINGS ==========
    logger.info("Loading BERT-TINY model (prajjwal1/bert-tiny) and encoding abstracts...")
    model_berttiny = SentenceTransformer('prajjwal1/bert-tiny')
    emb_train_berttiny = model_berttiny.encode(df_train['abstract'].tolist())
    emb_val_berttiny = model_berttiny.encode(df_validation['abstract'].tolist())
    emb_test_berttiny = model_berttiny.encode(df_test['abstract'].tolist())

    df_train['embeddings_berttiny'] = list(emb_train_berttiny)
    df_validation['embeddings_berttiny'] = list(emb_val_berttiny)
    df_test['embeddings_berttiny'] = list(emb_test_berttiny)
    logger.info(f"BERT-TINY embeddings shape: train {emb_train_berttiny.shape} | val {emb_val_berttiny.shape} | test {emb_test_berttiny.shape}")

    # ========== SAVE ==========
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_train.to_parquet(os.path.join(OUTPUT_DIR, 'NEW_OSHA_DATABASE_filtered_embeddings_train.parquet'), engine='pyarrow')
    df_validation.to_parquet(os.path.join(OUTPUT_DIR, 'NEW_OSHA_DATABASE_filtered_embeddings_val.parquet'), engine='pyarrow')
    df_test.to_parquet(os.path.join(OUTPUT_DIR, 'NEW_OSHA_DATABASE_filtered_embeddings_test.parquet'), engine='pyarrow')

    logger.info("Saved train, validation and test data to parquet files with TF-IDF, GEMMA, and BERT-TINY embeddings.")
    logger.info("===== STEP 1 COMPLETE =====")


if __name__ == '__main__':
    main()
