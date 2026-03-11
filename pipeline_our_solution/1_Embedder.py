import argparse
import logging
import os
from datetime import datetime

import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# -------------------------------
# CONSTANTS
# -------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 96
NUM_EPOCHS = 512
LEARNING_RATE = 1e-5
NUM_EPOCHS_BERT = 10
MAX_TOKENS = 512
LOG_DIR = "log"

SEVERITY_RANK = {
    "Fatality": 0,
    "Hospitalized injury": 1,
    "Non Hospitalized injury": 2,
    None: 3,
    float('nan'): 3
}
MIN_SEVERITY_RANK = 3

# link: https://huggingface.co/gaunernst/bert-small-uncased
BERT_MODELS = [
    "bert-base-uncased",  # Full:          12 layer, hidden 768, 110M parameters
]

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
def load_excel(file_path):
    try:
        logger.info(f"Loading Excel file: {file_path}")
        excel_data = pd.ExcelFile(file_path)
        logger.info(f"Sheets found: {excel_data.sheet_names}")
        return excel_data
    except Exception as e:
        logger.exception(f"Error loading Excel file: {e}")
        raise


def extract_dataset(excel_data, sheet_index):
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
# TOKENIZATION & FINE-TUNING
# -------------------------------
def tokenize_dataset(dataset, tokenizer, column_name):
    def tokenize_function(examples):
        return tokenizer(
            examples[column_name],
            truncation=True,
            padding="max_length",
            max_length=MAX_TOKENS
        )

    logger.info("Tokenizing dataset...")
    tokenized = dataset.map(tokenize_function, batched=True)
    logger.info("Tokenization complete.")
    return tokenized


def fine_tuning_bert_model(df_osha, df_aria, model_name=BERT_MODELS[0], save_model=True):
    logger.info(f"Preparing text data for BERT fine-tuning model {model_name}...")

    osha_texts = df_osha["abstract"].dropna().tolist()
    aria_texts = df_aria["Content"].dropna().tolist()
    all_texts = osha_texts + aria_texts

    logger.info(f"Total documents: {len(all_texts)}")
    dataset = Dataset.from_dict({"osha_aria_text": all_texts})

    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    model.to(DEVICE)

    tokenized_dataset = tokenize_dataset(dataset, tokenizer, column_name='osha_aria_text')

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=f"bert-mlm-finetuned-{model_name.replace('/', '_')}",
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS_BERT,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        do_eval=False,
        seed=RANDOM_SEED,
        logging_dir="log",
    )

    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    logger.info("Starting BERT fine-tuning...")
    trainer.train()
    logger.info("BERT fine-tuning complete.")

    if save_model:
        logger.info("Saving fine-tuned model...")
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info("Model saved successfully.")

    return model, tokenizer


# -------------------------------
# EMBEDDING EXTRACTION
# -------------------------------
def embedding_text(model, tokenizer, df, column_name, batch_size=64):
    logger.info(f"Embedding texts from column: {column_name}")
    model.to(DEVICE)
    model.eval()

    texts = df[column_name].dropna().tolist()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors='pt'
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        with torch.no_grad():
            model_output = model(**encoded, output_hidden_states=True)
            last_hidden_state = model_output.hidden_states[-1]
            embeddings = last_hidden_state[:, 0, :]
            all_embeddings.extend(embeddings.cpu().numpy())

    df['embeddings'] = all_embeddings
    logger.info(f"Generated embeddings for {len(all_embeddings)} texts.")


def combine_dataset(df_data, df_employee):
    """Perform a left join between df_data and df_employee on 'report_num',
    keeping the most severe 'Degree' in case of multiple matches."""
    try:
        logging.info("Preparing employee dataset for left join with 'max severity' logic...")

        df_employee["severity_level"] = df_employee["Degree"].map(lambda x: SEVERITY_RANK.get(x, MIN_SEVERITY_RANK))

        # For each report_num, keep the row with the minimum severity_level (i.e., most severe Degree)
        df_employee_reduced = df_employee.sort_values("severity_level").drop_duplicates(subset=["report_num"],
                                                                                        keep="first")

        logging.info(f"Reduced employee dataset to {df_employee_reduced.shape[0]} rows with most severe Degree.")

        # Perform left join
        df_merged = pd.merge(
            df_data,
            df_employee_reduced,
            on='report_num',
            how='left'
        )

        logging.info(f"Joined dataset contains {df_merged.shape[0]} rows and {df_merged.shape[1]} columns.")
        return df_merged

    except Exception as e:
        logging.error(f"Error during dataset combination: {e}")
        raise


# -------------------------------
# MAIN PIPELINE - 1 Embedder
# -------------------------------
def main(model_name):
    logger.info("===== STARTING PIPELINE - STEP 1 EMBEDDER =====")

    logging.info("\nStage 1: Loading datasets...")
    excel_osha = load_excel('../dataset/NEW_OSHA_DATABASE_FILTERED_SIC_NAICS.xlsx')
    df_data = extract_dataset(excel_osha, 0)
    df_employee = extract_dataset(excel_osha, 2)
    logging.info(f"OSHA dataset loaded with {df_data.shape[0]} rows and {df_data.shape[1]} columns.")

    df_aria = extract_dataset(load_excel('../dataset/aria_reports_final_clean.xlsx'), 0)
    logging.info(f"Aria dataset loaded with {df_aria.shape[0]} rows and {df_aria.shape[1]} columns.")

    df_osha = combine_dataset(df_data, df_employee)
    initial_len = len(df_osha)
    logging.info(f"Combined dataset with {df_osha.shape[0]} rows and {df_osha.shape[1]} columns.")

    logging.info(f"\nStage 2: Fine-tuning BERT model {model_name}...")
    model, tokenizer = fine_tuning_bert_model(df_osha,
                                              df_aria,
                                              model_name=model_name,
                                              save_model=False)

    logging.info(f"\nStage 3: Embedding texts from OSHA from column 'abstract'...")
    embedding_text(model, tokenizer, df_osha, 'abstract')

    logging.info(f"\nStage 4: Data reduction. Dropping rows with severity_level = 3...")
    df_osha.drop(df_osha[df_osha['severity_level'] == 3].index, inplace=True)
    dropped = initial_len - len(df_osha)
    logging.info(f"Rimosse {dropped} righe con severity_level = 4")

    logging.info(f"\nStage 5: Saving embeddings to parquet file...")
    os.makedirs('output', exist_ok=True)
    output_file = f"output/NEW_OSHA_DATABASE_FILTERED_SIC_NAICS_embeddings.parquet"
    df_osha.to_parquet(output_file, engine='pyarrow')
    logging.info(f"Parquet saved: {output_file}")

    logger.info("===== STEP 1 COMPLETE =====")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline di classificazione con modelli BERT")
    parser.add_argument('--model',
                        type=str,
                        default=BERT_MODELS[0],
                        help=f'Specifica il nome del modello BERT tra {BERT_MODELS}, default: {BERT_MODELS[0]}')
    args = parser.parse_args()

    model_name = args.model
    if model_name not in BERT_MODELS:
        logger.warning(f"Modello BERT '{model_name}' non valido, uso il modello di default: {BERT_MODELS[0]}")
        model_name = BERT_MODELS[0]

    main(model_name)
