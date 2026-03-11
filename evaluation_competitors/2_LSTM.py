import logging
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# -------- SEED TOTALE REPRODUCIBILITY --------
RANDOM_SEED = 96
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

# -------------------------------
# CONSTANTS
# -------------------------------
LOG_DIR = "log"
OUTPUT_DIR = "output"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

PAD_IDX = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# LOGGING CONFIGURATION
# -------------------------------
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"4_pytorch_lstm_reduced_tfidf_{timestamp}.log"
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


class LSTMSeverityClassifier(nn.Module):
    def __init__(self, input_dim=128, hidden_size=64, num_classes=3, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM per sequenze di embeddings numerici
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # LSTM: (batch_size, seq_len, hidden_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        out = self.dropout(hidden[-1])
        return self.fc(out)


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


def prepare_data(df, X_COL, Y_COL, train_mode=True):
    X = np.array([emb for emb in df[X_COL]], dtype=np.float32)
    y = df[Y_COL].values

    X = X.reshape(X.shape[0], 1, X.shape[1])

    if train_mode:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        joblib.dump(label_encoder, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        logger.info(
            f"Label encoder saved. Classes: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    else:
        label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        y_encoded = label_encoder.transform(y)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y_encoded)

    logger.info(f"Data prepared - X: {X_tensor.shape}, y: {y_tensor.shape}")
    return X_tensor, y_tensor, label_encoder


def train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=50):
    logger.info(f"Training LSTM on {DEVICE}...")
    logger.info(f"Model architecture:\n{model}")

    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train.to(DEVICE))
        loss = criterion(outputs, y_train.to(DEVICE))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.to(DEVICE))
            val_loss = criterion(val_outputs, y_val.to(DEVICE))
            val_losses.append(val_loss.item())

        logger.info(f"Epoch {epoch + 1}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

        # Early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    return model, train_losses, val_losses


def save_lstm_model(model, X_COL, model_name):
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': model.input_dim,
        'hidden_size': model.hidden_size,
        'num_classes': model.fc[-1].out_features,
        'feature_col': X_COL,
        'metadata': {
            'trained_at': datetime.now().isoformat(),
            'random_seed': RANDOM_SEED,
            'device': str(DEVICE),
            'architecture': 'LSTMSeverityClassifier'
        }
    }, model_path)
    logger.info(f"PyTorch LSTM model saved to: {model_path}")
    return model_path


def load_lstm_model(model_path, input_dim=128, hidden_size=64, num_classes=3):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = LSTMSeverityClassifier(
        input_dim=checkpoint['input_dim'],
        hidden_size=checkpoint['hidden_size'],
        num_classes=checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    logger.info(f"PyTorch LSTM model loaded from: {model_path}")
    return model


def evaluate_test_set(model, X_test, y_test, label_encoder):
    logger.info("Evaluating PyTorch LSTM on test set...")

    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(DEVICE))
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

    y_test_np = y_test.numpy()
    logger.info("\n" + "=" * 70)
    logger.info("TEST SET CLASSIFICATION REPORT - PYTORCH LSTM + REDUCED TF-IDF")
    logger.info("=" * 70)
    print(classification_report(y_test_np, y_pred, digits=4, zero_division=0,
                                target_names=[str(c) for c in label_encoder.classes_]))

    logger.info("\nConfusion Matrix:")
    logger.info(confusion_matrix(y_test_np, y_pred))

    return y_pred


def main():
    logger.info("===== STARTING PIPELINE - STEP 4 PYTORCH LSTM + REDUCED TF-IDF =====")
    logger.info(f"Using device: {DEVICE}")

    X_COL = 'embeddings_reduced_tfidf'
    Y_COL = 'severity_level'

    df_train, df_validation, df_test = load_parquet_datasets(X_COL, Y_COL)

    X_train, y_train, le = prepare_data(df_train, X_COL, Y_COL, train_mode=True)
    X_val, y_val, _ = prepare_data(df_validation, X_COL, Y_COL, train_mode=False)
    X_test, y_test, _ = prepare_data(df_test, X_COL, Y_COL, train_mode=False)

    input_dim = X_train.shape[2]  # 128
    model = LSTMSeverityClassifier(
        input_dim=input_dim,
        hidden_size=64,
        num_classes=len(le.classes_)
    )
    trained_model, train_losses, val_losses = train_lstm_model(model, X_train, y_train, X_val, y_val)

    model_path = save_lstm_model(trained_model, X_COL, "pytorch_lstm_reduced_tfidf")

    logger.info("Reloading PyTorch LSTM model from filesystem...")
    loaded_model = load_lstm_model(model_path)

    evaluate_test_set(loaded_model, X_test, y_test, le)

    logger.info("===== STEP COMPLETE =====")


if __name__ == '__main__':
    main()
