"""
Analyse Autoencodeur + SHAP avec reproductibilité (random seed)

- Autoencodeur fully connected
- Explication avec SHAP
- Reproductibilité assurée
"""

import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ==============================
# REPRODUCTIBILITÉ
# ==============================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # PyTorch deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================
# AUTOENCODER
# ==============================
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 10),
            nn.ReLU(),

            nn.Linear(10, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),

            nn.Linear(10, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


# ==============================
# DATA PREPROCESSING
# ==============================
def encode_yes_no_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        values = df[col].dropna()
        if df[col].dtype == object and values.isin(["Oui", "Non"]).all():
            df[col] = np.where(df[col] == "Oui", 1, 0).astype(np.int64)
    return df


def load_and_prepare_data(input_path: Path, sheet_name: str, target_column: str):
    df = pd.read_excel(input_path, sheet_name=sheet_name)

    if target_column in df.columns:
        df = df.drop(columns=[target_column])

    df = encode_yes_no_columns(df)

    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            "Colonnes non numériques détectées : "
            + ", ".join(non_numeric)
        )

    feature_names = df.columns.tolist()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    return x_scaled, x_tensor, feature_names


# ==============================
# TRAINING
# ==============================
def train_autoencoder(x_tensor, epochs, batch_size, lr, seed):

    input_dim = x_tensor.shape[1]
    model = Autoencoder(input_dim)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # IMPORTANT: reproductibilité du DataLoader
    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        TensorDataset(x_tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator
    )

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for (x_batch,) in loader:
            reconstructed, _ = model(x_batch)

            loss = criterion(reconstructed, x_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(loader):.6f}")

    return model


# ==============================
# SHAP ANALYSIS
# ==============================
def compute_shap_interactions(model, x_tensor, feature_names, n_samples):

    model.eval()
    n_samples = min(n_samples, x_tensor.shape[0])

    background = x_tensor[:n_samples]
    explained = x_tensor[:n_samples]

    explainer = shap.DeepExplainer(model.encoder, background)
    shap_values = explainer.shap_values(explained)

    # Agrégation multi-dimension latente
    if isinstance(shap_values, list):
        shap_array = np.stack(shap_values, axis=-1)
        shap_global = np.mean(np.abs(shap_array), axis=-1)
    else:
        shap_array = np.asarray(shap_values)
        shap_global = np.mean(np.abs(shap_array), axis=-1)

    interaction_scores = {}
    n_features = len(feature_names)

    for i in range(n_features):
        for j in range(i + 1, n_features):
            name = f"{feature_names[i]} + {feature_names[j]}"
            score = float(np.mean(shap_global[:, i] + shap_global[:, j]))
            interaction_scores[name] = score

    sorted_interactions = sorted(
        interaction_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    df_all = pd.DataFrame(
        sorted_interactions,
        columns=["Combinaison", "Impact SHAP"]
    )

    df_all["Rang"] = range(1, len(df_all) + 1)

    return df_all


# ==============================
# MAIN
# ==============================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--sheet", default="témoins")
    parser.add_argument("--target-column", default="Variable dépendante")
    parser.add_argument("--output", required=True, type=Path)

    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--shap-samples", default=100, type=int)

    # nouveau
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    # appliquer seed
    set_seed(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    x_scaled, x_tensor, feature_names = load_and_prepare_data(
        args.input,
        args.sheet,
        args.target_column
    )

    model = train_autoencoder(
        x_tensor,
        args.epochs,
        args.batch_size,
        args.lr,
        args.seed
    )

    df = compute_shap_interactions(
        model,
        x_tensor,
        feature_names,
        args.shap_samples
    )

    df.head(10).to_csv(args.output, index=False)

    print("Top 10 sauvegardé :", args.output)
    print(df.head(10))


if __name__ == "__main__":
    main()
