import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_curve, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.model import Autoencoder

SEED = 42
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data():
    X_train_licit = np.load(PROCESSED_DATA_PATH / "X_train_licit.npy")
    X_val = np.load(PROCESSED_DATA_PATH / "X_val.npy")
    X_test = np.load(PROCESSED_DATA_PATH / "X_test.npy")
    y_val = np.load(PROCESSED_DATA_PATH / "y_val.npy")
    y_test = np.load(PROCESSED_DATA_PATH / "y_test.npy")
    scaler = joblib.load(PROCESSED_DATA_PATH / "scaler.joblib")
    return X_train_licit, X_val, X_test, y_val, y_test, scaler


def make_loader(x_train, batch_size):
    tensor = torch.tensor(x_train, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model(model, train_loader, device, lr, epochs, weight_decay):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for (batch,) in train_loader:
            inputs = batch.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        losses.append(epoch_loss)
    return model, losses


def evaluate_model(model, X_val, X_test, y_val, y_test, scaler, device):
    with torch.no_grad():
        val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        val_recon = model(val_tensor).cpu().numpy()
        test_recon = model(test_tensor).cpu().numpy()

    X_val_orig = scaler.inverse_transform(X_val)
    X_test_orig = scaler.inverse_transform(X_test)
    X_val_rec_orig = scaler.inverse_transform(val_recon)
    X_test_rec_orig = scaler.inverse_transform(test_recon)

    val_errors = np.mean(np.square(X_val_orig - X_val_rec_orig), axis=1)
    test_errors = np.mean(np.square(X_test_orig - X_test_rec_orig), axis=1)

    precision, recall, thresholds = precision_recall_curve(y_val, val_errors)
    precision_valid = precision[1:]
    recall_valid = recall[1:]
    f1_scores = 2 * (precision_valid * recall_valid) / (precision_valid + recall_valid + 1e-10)
    best_idx = int(np.argmax(f1_scores))
    optimal_threshold = thresholds[best_idx]
    y_pred = (test_errors >= optimal_threshold).astype(int)

    metrics = {
        "optimal_threshold": float(optimal_threshold),
        "val_best_f1": float(f1_scores[best_idx]),
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_avg_precision": float(average_precision_score(y_test, test_errors)),
        "test_roc_auc": float(roc_auc_score(y_test, test_errors)),
        "test_confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics, test_errors, val_errors


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train_licit, X_val, X_test, y_val, y_test, scaler = load_data()
    input_dim = X_train_licit.shape[1]

    configs = [
        {"name": "lr1e-3_ep30_bn8_h32_16_do0", "lr": 1e-3, "epochs": 30, "bottleneck_dim": 8, "hidden_dims": (32, 16), "dropout_rate": 0.0, "weight_decay": 1e-5, "batch_size": 256},
        {"name": "lr1e-3_ep40_bn16_h64_32_do0", "lr": 1e-3, "epochs": 40, "bottleneck_dim": 16, "hidden_dims": (64, 32), "dropout_rate": 0.0, "weight_decay": 1e-5, "batch_size": 256},
        {"name": "lr5e-4_ep40_bn16_h64_32_do1", "lr": 5e-4, "epochs": 40, "bottleneck_dim": 16, "hidden_dims": (64, 32), "dropout_rate": 0.1, "weight_decay": 1e-5, "batch_size": 256},
        {"name": "lr1e-4_ep60_bn16_h64_32_do0", "lr": 1e-4, "epochs": 60, "bottleneck_dim": 16, "hidden_dims": (64, 32), "dropout_rate": 0.0, "weight_decay": 1e-5, "batch_size": 256},
    ]

    results = []
    best_result = None
    best_model_state = None

    for cfg in configs:
        print(f"\n=== Running {cfg['name']} ===")
        model = Autoencoder(
            input_dim=input_dim,
            bottleneck_dim=cfg["bottleneck_dim"],
            hidden_dims=cfg["hidden_dims"],
            dropout_rate=cfg["dropout_rate"],
        ).to(device)
        train_loader = make_loader(X_train_licit, cfg["batch_size"])
        model, losses = train_model(model, train_loader, device, cfg["lr"], cfg["epochs"], cfg["weight_decay"])
        metrics, _, _ = evaluate_model(model, X_val, X_test, y_val, y_test, scaler, device)
        summary = {"name": cfg["name"], "config": cfg, "metrics": metrics, "final_train_loss": float(losses[-1]) if losses else None}
        results.append(summary)
        print(json.dumps(summary["metrics"], indent=2))

        if best_result is None or summary["metrics"]["test_avg_precision"] > best_result["metrics"]["test_avg_precision"]:
            best_result = summary
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_model_state is not None:
        torch.save(best_model_state, MODELS_DIR / "best_autoencoder_hparam_search.pth")

    with open(MODELS_DIR / "hyperparameter_search_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nBEST RESULT")
    print(json.dumps(best_result, indent=2))


if __name__ == "__main__":
    main()
