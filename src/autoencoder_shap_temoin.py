"""
Analyse Autoencodeur + SHAP sur les cas témoins.

Ce script charge une feuille Excel, prépare les variables, entraîne un autoencodeur
non supervisé et applique SHAP sur l'encodeur afin d'identifier les variables qui
structurent le plus la représentation latente.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


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
            nn.Linear(10, 3),
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
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


def encode_yes_no_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Encode les colonnes contenant uniquement Oui/Non en 1/0."""
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

    # Sécurité : garder uniquement les colonnes numériques après encodage.
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            "Colonnes non numériques détectées après encodage : "
            + ", ".join(non_numeric)
            + ". Ajouter un encodage adapté avant l'entraînement."
        )

    feature_names = df.columns.tolist()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    return x_scaled, x_tensor, feature_names


def train_autoencoder(x_tensor: torch.Tensor, epochs: int, batch_size: int, lr: float):
    input_dim = x_tensor.shape[1]
    model = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=True, drop_last=True)

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
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / max(len(loader), 1):.6f}")

    return model


def compute_shap_interactions(model: Autoencoder, x_tensor: torch.Tensor, x_scaled, feature_names, n_samples: int):
    model.eval()
    n_samples = min(n_samples, x_tensor.shape[0])

    background = x_tensor[:n_samples]
    explained = x_tensor[:n_samples]

    explainer = shap.DeepExplainer(model.encoder, background)
    shap_values = explainer.shap_values(explained)

    # shap_values peut être une liste si la sortie latente est multi-dimensionnelle.
    # On agrège les dimensions latentes pour obtenir une importance globale.
    if isinstance(shap_values, list):
        shap_array = np.stack(shap_values, axis=-1)
        shap_global = np.mean(np.abs(shap_array), axis=-1)
    else:
        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            shap_global = np.mean(np.abs(shap_array), axis=-1)
        else:
            shap_global = np.abs(shap_array)

    interaction_scores = {}
    n_features = len(feature_names)

    for i in range(n_features):
        for j in range(i + 1, n_features):
            name = f"{feature_names[i]} + {feature_names[j]}"
            score = float(np.mean(shap_global[:, i] + shap_global[:, j]))
            interaction_scores[name] = score

    sorted_interactions = sorted(interaction_scores.items(), key=lambda x: x[1], reverse=True)
    df_all = pd.DataFrame(sorted_interactions, columns=["Combinaison de facteurs", "Impact moyen SHAP"])
    df_all["Rang"] = range(1, len(df_all) + 1)
    return df_all


def main():
    parser = argparse.ArgumentParser(description="Autoencodeur + SHAP pour les cas témoins")
    parser.add_argument("--input", required=True, type=Path, help="Chemin du fichier Excel")
    parser.add_argument("--sheet", default="témoins", help="Nom de la feuille Excel")
    parser.add_argument("--target-column", default="Variable dépendante", help="Colonne cible à supprimer")
    parser.add_argument("--output", required=True, type=Path, help="Chemin du CSV de sortie")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--shap-samples", default=100, type=int)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    x_scaled, x_tensor, feature_names = load_and_prepare_data(args.input, args.sheet, args.target_column)
    model = train_autoencoder(x_tensor, args.epochs, args.batch_size, args.lr)
    df_interactions = compute_shap_interactions(model, x_tensor, x_scaled, feature_names, args.shap_samples)

    df_interactions.head(10).to_csv(args.output, index=False)
    print("Top 10 combinaisons sauvegardées dans :", args.output)
    print(df_interactions.head(10))


if __name__ == "__main__":
    main()
