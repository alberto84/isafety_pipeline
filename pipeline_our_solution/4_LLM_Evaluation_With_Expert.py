import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from ollama import Client
from pydantic import BaseModel
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import ollama

SYSTEM_PROMPT = """
You are an expert language model specialized in analyzing injury and incident reports. Your task is to classify user-provided incident descriptions into one of the following three categories: "Hospitalized injury", "Non Hospitalized injury", or "Fatality".

**Classification Definitions:**
-   **Fatality:** The incident resulted in the death of one or more individuals.
-   **Hospitalized injury:** The incident resulted in an injury requiring formal *in-patient admission* to a hospital or clinic. This means the person was admitted and stayed for treatment. Treatment only in an Emergency Room (ER) does *not* count as hospitalization.
-   **Non Hospitalized injury:** The injury occurred but did *not* require in-patient admission. This includes cases treated on-site, in outpatient settings, or in the Emergency Room without being admitted.

**Injury Type Context (for understanding, not direct classification):**
The following provides context on various injury types. Use this to better understand the nature of the described incident, but remember that the *outcome* (hospitalization, non-hospitalization, fatality) is paramount, not just the type of injury.

-   **(BONE) FRACTURES:** Breaks or chips of bone or teeth (e.g., fractures, broken bones, cracked bones, broken or chipped tooth, joint fractures, back fractures, neck fractures).
-   **BURNS:** Tissue damage from heat, flame, chemicals, electricity, cold, etc. (e.g., heat burns, chemical burns, electrical burns, road rash, Welder’s flash).
-   **AMPUTATIONS:** Traumatic severing or loss of a body part (e.g., amputations, traumatic loss, bone loss, crushing injuries, avulsions).
-   **TRAUMATIC INJURIES TO MUSCLES, TENDONS, LIGAMENTS, JOINTS, ETC.:** Injuries affecting these body parts (e.g., dislocations, torn cartilage, herniated disc, sprains, strains, whiplash).
-   **OPEN WOUNDS (NO AMPUTATIONS):** Injuries with broken skin (e.g., laceration, cut, puncture).
-   **ELECTROCUTION:** Injuries from contact with electricity (e.g., electric shocks, arc flashes, contact with power line).
-   **INJURIES FROM HARMFUL SUBSTANCES OR PHYSICAL AGENTS:** Injuries from exposure to chemicals, radiation, noise, extreme temperatures, etc. (e.g., cancer, solvents, asphyxia, radiation, hearing loss, heat stress, cold stress).

**Guidelines for Classification:**
1.  **Prioritize Outcome:** Your primary focus is on the *outcome* of the incident (death, in-patient admission, or no in-patient admission), not just the type or severity of the injury itself.
2.  **Explicit Evidence:** Only classify as "Fatality" or "Hospitalized injury" if the description explicitly states or strongly implies death or in-patient admission. Do not assume.
3.  **Severity vs. Outcome:** A severe injury (like a broken arm or finger amputation) does not automatically mean "Hospitalized injury" unless *in-patient admission* is clearly indicated. Without such indication, even severe injuries should default to "Non Hospitalized injury" if no fatality is mentioned.

Your response must be a valid JSON object containing:
-   `hospitalized_injury`: integer percentage probability.
-   `non_hospitalized_injury`: integer percentage probability.
-   `fatality`: integer percentage probability.
-   `explanation`: a string explaining your classification.
"""

USER_PROMPT = """
The following examples classify a text description based on the definitions and guidelines provided:

{samples}

Now, classify the following incident description:

{text}

Provide your response as a valid JSON object with percentage probabilities for each class and an explanation.
"""

# -------------------------------
# CONSTANTS
# -------------------------------
RANDOM_SEED = 96
NUM_SEVERITY_LEVELS = 3  # 0,1,2
LOG_DIR = "log"

# -------------------------------
# LOGGING CONFIGURATION
# -------------------------------
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"4_llm_evaluation_with_expert_{timestamp}.log"
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

client = Client(host='http://localhost:11434')


class ClsResponse(BaseModel):
    hospitalized_injury: int
    non_hospitalized_injury: int
    fatality: int
    explanation: str


def make_call_to_llm(abstract, result_str):
    response = ollama.chat(
        model='gemma2',
        format=ClsResponse.model_json_schema(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {'role': 'user', 'content': USER_PROMPT.format(samples=result_str, text=abstract)},
        ])

    cls_response = ClsResponse.model_validate_json(response['message']['content'])
    return cls_response


def sample_per_clusters(df_cluster, num_samples=5):
    unique_clusters = df_cluster['cluster'].unique()
    severity_levels = range(0, NUM_SEVERITY_LEVELS)  # 0,1,2

    sampled_dfs = []

    for cluster_id in unique_clusters:
        for severity in severity_levels:
            df_filtered = df_cluster[
                (df_cluster['cluster'] == cluster_id) &
                (df_cluster['severity_level'] == severity)
                ]

            if df_filtered.empty:
                df_candidates = df_cluster[df_cluster['severity_level'] == severity]
                logger.info(
                    f"No samples in cluster {cluster_id}, severity {severity}; using global ({len(df_candidates)})")
                if df_candidates.empty:
                    logger.warning(f"No samples for severity level {severity}")
                    continue
            else:
                df_candidates = df_filtered

            sampled = df_candidates.sample(min(num_samples, len(df_candidates)), random_state=RANDOM_SEED)
            sampled_dfs.append(sampled)

    if sampled_dfs:
        return pd.concat(sampled_dfs, ignore_index=True).drop_duplicates()
    return pd.DataFrame()


def process_data(df_data, df_samples, column_name):
    severity_levels = set(range(0, NUM_SEVERITY_LEVELS))  # 0,1,2

    for index, row in df_data.iterrows():
        cluster = row['cluster']
        df_sample_cluster = df_samples[df_samples['cluster'] == cluster]

        result_str = ''

        for severity in severity_levels:
            df_severity = df_sample_cluster[df_sample_cluster['severity_level'] == severity]
            if not df_severity.empty:
                sample = df_severity.sample(1).iloc[0]
            else:
                df_severity_global = df_samples[df_samples['severity_level'] == severity]
                if not df_severity_global.empty:
                    sample = df_severity_global.sample(1).iloc[0]
                else:
                    continue

            result_str += f'"{sample["abstract"]}"\nCLASS: {sample["Degree"]}\n\n'

        if result_str:
            try:
                cls_response = make_call_to_llm(row[column_name], result_str)
                df_data.at[index, 'hospitalized_injury'] = cls_response.hospitalized_injury
                df_data.at[index, 'non_hospitalized_injury'] = cls_response.non_hospitalized_injury
                df_data.at[index, 'fatality'] = cls_response.fatality
                df_data.at[index, 'explanation'] = cls_response.explanation
                df_data.at[index, 'system_prompt'] = SYSTEM_PROMPT
                df_data.at[index, 'user_prompt'] = USER_PROMPT.format(samples=result_str, text=row[column_name])
            except Exception as e:
                logger.error(f"LLM failed row {index}: {e}")
                df_data.at[index, 'hospitalized_injury'] = 33
                df_data.at[index, 'non_hospitalized_injury'] = 33
                df_data.at[index, 'fatality'] = 34
                df_data.at[index, 'explanation'] = "LLM failed"
                df_data.at[index, 'user_prompt'] = "LLM failed"


def calculate_pred(row):
    """0=Fatality, 1=Hospitalized, 2=Non-hospitalized"""
    columns = ['fatality', 'hospitalized_injury', 'non_hospitalized_injury']
    values = [row[col] for col in columns]
    max_index = values.index(max(values))
    return max_index  # Returns 0,1,2


def save_results_to_file(output_file, cl_report_dict, desc_dict=None):
    report_dict = {}
    for key, value in cl_report_dict.items():
        if isinstance(value, dict):
            for k2, v in value.items():
                report_dict[f'{key}_{k2}'] = [v]
        else:
            report_dict[key] = [value]

    output_file = Path(output_file)
    if desc_dict:
        report_dict.update(desc_dict)
        columns = list(desc_dict.keys()) + sorted([k for k in report_dict if k not in desc_dict])
    else:
        columns = sorted(report_dict.keys())

    df_log = pd.DataFrame({k: [v] for k, v in report_dict.items()})[columns]

    if output_file.exists():
        df_existing = pd.read_excel(output_file)
        df_log = pd.concat([df_existing, df_log], ignore_index=True)

    df_log.to_excel(output_file, index=False)


def evaluate_classification_metrics(df, plot_confusion_matrix=True):
    """Evaluate test set metrics"""
    df = df.copy()

    label_mapping = {
        0: "Fatality",
        1: "Hospitalized injury",
        2: "Non Hospitalized injury"
    }

    y_true = df['severity_level'].map(label_mapping)
    y_pred = df['y_pred'].map(label_mapping)

    metrics = {
        "macro_precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "micro_precision": precision_score(y_true, y_pred, average='micro', zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average='micro', zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average='micro', zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    logger.info("\n=== TEST SET RESULTS ===")
    results = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    desc_dict = {
        'dataset': 'test',
        'n_samples': len(df),
        'severity_mapping': '0=Fatality,1=Hospitalized,2=NonHospitalized',
        'timestamp': datetime.now().isoformat()
    }

    save_results_to_file('output/classification_report_test.xlsx', results, desc_dict)

    logger.info("Classification Report saved")
    logger.info(results)

    logger.info("\nPrecision/Recall/F1:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.3f}")

    # Confusion Matrix
    labels_order = ["Fatality", "Hospitalized injury", "Non Hospitalized injury"]
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    cm_df = pd.DataFrame(cm, index=labels_order, columns=labels_order)

    logger.info(f"\nConfusion Matrix:\n{cm_df}")

    if plot_confusion_matrix:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix - Test Set (0-1-2 mapping)')
        plt.tight_layout()
        plt.savefig('output/confusion_matrix_test.png', dpi=300, bbox_inches='tight')
        plt.show()

    return metrics


def main():
    logger.info("===== STEP 4 LLM EVALUATION (TEST ONLY - 0/1/2) =====")

    # Load test set
    df_test = pd.read_parquet('output/cluster_test.parquet', engine='pyarrow')
    df_test = df_test.drop(columns=['embeddings', 'embeddings_reduced'])
    logger.info(f"Test set loaded: {df_test.shape}")
    logger.info(f"Severity levels: {sorted(df_test['severity_level'].unique())}")

    # Sample for few-shot
    samples = sample_per_clusters(df_test, num_samples=5)
    logger.info(f"Few-shot samples: {len(samples)}")

    # Add prediction columns
    df_test['hospitalized_injury'] = 0
    df_test['non_hospitalized_injury'] = 0
    df_test['fatality'] = 0
    df_test['explanation'] = ''
    df_test['user_prompt'] = ''
    df_test['system_prompt'] = ''

    # LLM predictions
    process_data(df_test, samples, 'abstract')

    # Calculate final predictions
    df_test['y_pred'] = df_test.apply(calculate_pred, axis=1)

    # Save full results
    df_test.to_parquet('output/test_llm_predictions.parquet', engine='pyarrow', index=False)
    logger.info("Saved: test_llm_predictions.parquet")

    # Evaluate
    metrics = evaluate_classification_metrics(df_test)

    # Metrics summary
    metrics_summary = {'dataset': 'test', **metrics}
    pd.DataFrame([metrics_summary]).to_csv('output/test_metrics.csv', index=False)


    df_filtered = df_test[['report_num', 'Report ID', 'severity_level', 'system_prompt', 'user_prompt']].copy()
    df_filtered.to_parquet('output/test_dataset_with_expert.parquet', engine='pyarrow', index=False)

    logger.info("===== COMPLETE =====")


if __name__ == '__main__':
    main()
