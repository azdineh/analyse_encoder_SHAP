# Analyse - Autoencodeur et SHAP pour données biomédicales

Ce dépôt documente une analyse exploratoire menée sur un jeu de données biomédical relatif au cancer du sein. L'étude utilise un autoencodeur pour apprendre une représentation latente des profils des individus témoins, puis SHAP pour interpréter les variables qui contribuent le plus à cette représentation.

## Objectif scientifique

L'objectif principal est d'identifier les facteurs qui structurent les profils des cas témoins à partir d'un modèle non supervisé. Les résultats ne doivent pas être interprétés comme des facteurs causaux ou prédictifs de la pathologie, mais comme des variables influentes dans la représentation latente apprise par l'autoencodeur.

## Données

Le fichier attendu est :

```text
data/data_deep_learning_sein.xlsx
```

La feuille utilisée dans l'expérience est :

```text
témoins
```

La colonne `Variable dépendante` est supprimée, car l'analyse est réalisée sans variable cible dans le cadre non supervisé.

## Pipeline méthodologique

1. Chargement des données depuis Excel.
2. Sélection de la feuille `témoins`.
3. Suppression de la variable dépendante.
4. Encodage des variables binaires `Oui` / `Non` en `1` / `0`.
5. Standardisation des variables avec `StandardScaler`.
6. Entraînement d'un autoencodeur profond.
7. Extraction de la représentation latente par l'encodeur.
8. Application de SHAP sur l'encodeur.
9. Classement des variables et combinaisons de facteurs selon leur impact moyen SHAP.

## Structure du dépôt

```text
analyse-hajuba-github/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── src/
│   └── autoencoder_shap_temoin.py
├── docs/
│   ├── methodologie.md
│   ├── interpretation_autoencoder_shap.md
│   ├── references.md
│   └── analyse-hajiba-original.docx
├── data/
│   └── README.md
├── results/
│   └── README.md
└── notebooks/
    └── README.md
```

## Exécution

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Lancer l'analyse :

```bash
python src/autoencoder_shap_temoin.py \
  --input data/data_deep_learning_sein.xlsx \
  --sheet "témoins" \
  --output results/top_10_combinaisons_SHAP_sein_temoins_rang.csv
```

## Interprétation attendue

Les valeurs SHAP indiquent les variables qui influencent le plus la représentation latente apprise par l'autoencodeur. Elles ne démontrent pas une relation directe avec le cancer du sein, car l'entraînement est effectué uniquement sur les cas témoins et sans variable cible.

## Références scientifiques principales

- Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- Antwarg, L., Miller, R. M., Shapira, B., & Rokach, L. (2021). *Explaining anomalies detected by autoencoders using SHAP*. Expert Systems with Applications.
- Molnar, C. (2022). *Interpretable Machine Learning*.
