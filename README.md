# Large Language Models for Occupational Accident Analysis: Classification, Explainability, and Data Enrichment

This repository contains the Python prototype used in the paper **"Large Language Models for Occupational Accident Analysis: Classification, Explainability, and Data Enrichment"**.

The project implements a multi-stage pipeline for occupational accident report analysis. It processes free-text accident narratives, generates semantic representations, groups similar reports, and uses a local Large Language Model (LLM) to classify accident severity while producing human-readable explanations. The same framework can also support data enrichment by inferring missing severity labels across heterogeneous accident-report sources.

## Paper Overview

The framework addresses severity classification for workplace accident reports using three target classes:

- `Fatality`
- `Hospitalized injury`
- `Non Hospitalized injury`

The proposed solution defines a pipeline, that includes:

- a domain-adapted `bert-base-uncased` embedder;
- an autoencoder-based dimensionality reduction module;
- K-Means clustering over reduced semantic representations;
- cluster-aware few-shot prompting;
- local LLM inference with Gemma 2 through Ollama;
- optional domain expert knowledge in the system prompt;
- probability-based severity prediction and textual explanations.

In the experiments reported in the paper, the proposed framework achieved an accuracy of `0.860` and a macro-F1 score of `0.836`, outperforming the evaluated traditional ML, neural, and literature-inspired baselines.

## Repository Structure

The repository includes the following files and folders:

- `pipeline_our_solution/`: implementation of the proposed framework.
- `pipeline_our_solution/1_Embedder.py`: fine-tunes BERT on OSHA and ARIA reports and creates the accident report embeddings.
- `pipeline_our_solution/2_Data_Reduction.py`: trains the autoencoder and produces reduced embeddings.
- `pipeline_our_solution/3_Clustering.py`: performs K-Means clustering and creates train, validation, and test splits with cluster assignments.
- `pipeline_our_solution/4_LLM_Evaluation_With_Expert.py`: evaluates Gemma 2 with domain expert knowledge in the prompt.
- `pipeline_our_solution/4_LLM_Evaluation_Without_Expert.py`: evaluates Gemma 2 without domain expert knowledge for ablation analysis.
- `evaluation_competitors/`: scripts used to reproduce competitor and baseline methods.
- `evaluation_competitors/1_Embedder.py`: creates TF-IDF, reduced TF-IDF, Gemma, and BERT-Tiny embeddings for baseline evaluation.
- `evaluation_competitors/2_SVM.py`: SVM baseline.
- `evaluation_competitors/2_XGboost.py`: XGBoost baseline.
- `evaluation_competitors/2_LSTM.py`: PyTorch LSTM baseline.
- `evaluation_competitors/2_KMEANS.py`: K-Means baseline.
- `README.md`: this file.

Generated outputs are written to local `output/` folders inside the corresponding pipeline directories. Runtime logs are written to local `log/` folders.

## Data

The raw datasets are not included in this repository. The scripts expect a `dataset/` folder at the repository root:

```text
isafety_pipeline-main/
|-- dataset/
|   |-- NEW_OSHA_DATABASE_FILTERED_SIC_NAICS.xlsx
|   `-- aria_reports_final_clean.xlsx
|-- pipeline_our_solution/
`-- evaluation_competitors/
```

Expected input files:

- `dataset/NEW_OSHA_DATABASE_FILTERED_SIC_NAICS.xlsx`
  - sheet `0`: OSHA accident reports, including the textual accident description in the `abstract` column;
  - sheet `2`: employee-level information, including `report_num` and `Degree`.
- `dataset/aria_reports_final_clean.xlsx`
  - sheet `0`: ARIA accident reports, including the accident text in the `Content` column.

The OSHA event-level severity is derived from the most severe employee-level `Degree` associated with each `report_num`.

## Environment Setup

The prototype was tested in the paper with Python `3.13`, PyTorch `2.7`, and Transformers `4.52.4`.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the required Python packages:

```bash
python -m pip install pandas numpy torch transformers datasets tqdm scikit-learn scipy pyclustering joblib pyarrow matplotlib seaborn ollama pydantic sentence-transformers xgboost openpyxl
```

For GPU-enabled PyTorch installations, follow the official PyTorch installation command matching your CUDA version.

## Ollama Setup

The LLM evaluation scripts call a local Ollama server at `http://localhost:11434` and use the `gemma2` model.

Install Ollama and pull the model:

```bash
ollama pull gemma2
```

Start the Ollama server before running the LLM evaluation scripts:

```bash
ollama serve
```

If Ollama is already running as a service, this step can be skipped.

## Running the Proposed Pipeline

Run the scripts from inside `pipeline_our_solution/`, because the paths in the code are relative to that directory.

```bash
cd pipeline_our_solution
```

1. Generate BERT embeddings:

```bash
python 1_Embedder.py --model bert-base-uncased
```

This produces:

```text
output/NEW_OSHA_DATABASE_FILTERED_SIC_NAICS_embeddings.parquet
```

2. Reduce embeddings with the autoencoder:

```bash
python 2_Data_Reduction.py --latent_dim 128
```

This produces:

```text
output/NEW_OSHA_DATABASE_FILTERED_SIC_NAICS_embeddings_reduced.parquet
```

3. Cluster reports and create train, validation, and test partitions:

```bash
python 3_Clustering.py --num_clusters 4
```

This produces:

```text
output/cluster_kmeans.joblib
output/cluster_centers.npy
output/cluster_train.parquet
output/cluster_validation.parquet
output/cluster_test.parquet
```

4. Evaluate the LLM-based classifier with domain expert knowledge:

```bash
python 4_LLM_Evaluation_With_Expert.py
```

5. Optionally run the ablation without domain expert knowledge:

```bash
python 4_LLM_Evaluation_Without_Expert.py
```

Both evaluation scripts write metrics and predictions to `output/`. If you run both configurations, copy or rename the common output files between runs if you need to keep both full result sets.

Main evaluation outputs include:

- `output/test_llm_predictions.parquet`
- `output/test_metrics.csv`
- `output/classification_report_test.xlsx`
- `output/confusion_matrix_test.png`
- `output/test_dataset_with_expert.parquet`
- `output/test_dataset_without_expert.parquet`

## Running Competitor Models

Run these scripts from inside `evaluation_competitors/`:

```bash
cd evaluation_competitors
```

First generate the baseline embeddings:

```bash
python 1_Embedder.py
```

Then run the competitor models:

```bash
python 2_SVM.py
python 2_XGboost.py
python 2_LSTM.py
python 2_KMEANS.py
```

The competitor embedder uses `google/embeddinggemma-300m`. If Hugging Face requires authentication for your environment, replace the placeholder token in `evaluation_competitors/1_Embedder.py` with your token or configure authentication through the Hugging Face CLI.

## Output Folders

Each pipeline folder creates its own runtime directories:

- `log/`: timestamped execution logs.
- `output/`: generated parquet files, trained models, metrics, Excel reports, and figures.
- `output/models/`: serialized baseline and competitor models.

These generated files can be large and are normally excluded from version control.

## Reproducibility Notes

- The scripts use `RANDOM_SEED` where applicable.
- The main evaluation follows a hold-out protocol with train, validation, and test partitions.
- In the paper, the final setting uses `4` clusters and a `128`-dimensional autoencoder latent space.
- Severity labels are mapped internally as:

```text
0 = Fatality
1 = Hospitalized injury
2 = Non Hospitalized injury
```

## Troubleshooting

- If a script cannot find the Excel files, verify that `dataset/` is at the repository root and that you are running the script from the correct pipeline directory.
- If the LLM evaluation fails, verify that Ollama is running and that `gemma2` is available locally.
- If `pyarrow` errors occur while saving parquet files, reinstall or upgrade `pyarrow`.
- If GPU memory is limited, reduce batch sizes in the embedding scripts or run on CPU.

## Citation

If you use this code, please cite the associated paper:

```bibtex
@article{cocca2026llmaccidentanalysis,
  author  = {Cocca, Paola and Falcone, Alberto and Guarascio, Massimo and Pisani, Francesco Sergio and Stefana, Elena and Tomasoni, Giuseppe},
  title   = {{Large Language Models for Occupational Accident Analysis: Classification, Explainability, and Data Enrichment}},
  journal = {IEEE Access},
  year    = {2026},
  note    = {in press}
}
```

Please update the citation with volume, pages, and DOI when they become available.
