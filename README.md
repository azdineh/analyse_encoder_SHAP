# Analysis - Autoencoder and SHAP for Biomedical Data

This repository documents an exploratory analysis conducted on a biomedical dataset related to breast cancer. The study uses an autoencoder to learn a latent representation of control individuals, and SHAP to interpret the variables that contribute most to this representation.

## Scientific Objective

The main objective is to identify the factors that structure the profiles of control cases using an unsupervised model. The results should not be interpreted as causal or predictive factors of the disease, but rather as influential variables in the latent representation learned by the autoencoder.

## Data

The expected file is:

data/data_deep_learning_sein.xlsx

The sheet used in the experiment is:

témoins

The "Dependent Variable" column is removed, as the analysis is performed without a target variable in an unsupervised setting.

## Methodological Pipeline

1. Load data from Excel.
2. Select the "témoins" sheet.
3. Remove the dependent variable.
4. Encode binary variables "Yes" / "No" as 1 / 0.
5. Standardize features using StandardScaler.
6. Train a deep autoencoder.
7. Extract the latent representation using the encoder.
8. Apply SHAP to the encoder.
9. Rank variables and factor combinations based on their mean SHAP impact.

## Repository Structure

analyse-hajuba-github/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── src/
│   └── autoencoder_shap_temoin.py
├── docs/
│   ├── methodology.md
│   ├── interpretation_autoencoder_shap.md
│   ├── references.md
│   └── analyse-hajiba-original.docx
├── data/
│   └── README.md
├── results/
│   └── README.md
└── notebooks/
    └── README.md

## Execution

Install dependencies:

pip install -r requirements.txt

Run the analysis:

python src/autoencoder_shap_temoin.py \
  --input data/data_deep_learning_sein.xlsx \
  --sheet "témoins" \
  --output results/top_10_combinaisons_SHAP_sein_temoins_rang.csv

## Expected Interpretation

SHAP values indicate which variables most influence the latent representation learned by the autoencoder. They do not demonstrate a direct relationship with breast cancer, as the model is trained only on control cases and without a target variable.

## Key Scientific References

- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
- Antwarg, L., Miller, R. M., Shapira, B., & Rokach, L. (2021). Explaining anomalies detected by autoencoders using SHAP. Expert Systems with Applications.
- Molnar, C. (2022). Interpretable Machine Learning.
