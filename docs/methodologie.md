# Méthodologie

## 1. Préparation des données

L'analyse porte sur la feuille `témoins` du fichier Excel. La variable dépendante est supprimée afin de conserver uniquement les facteurs explicatifs. Cette décision est cohérente avec l'objectif de l'étude, qui consiste à explorer la structure interne des profils témoins sans supervision par un label de maladie.

Les variables binaires codées sous forme textuelle (`Oui`, `Non`) sont transformées en variables numériques :

- `Oui` devient `1`
- `Non` devient `0`

Cette étape est indispensable pour rendre les données compatibles avec les modèles d'apprentissage profond.

## 2. Standardisation

Les variables sont standardisées avec `StandardScaler`. La standardisation permet de centrer et réduire les variables afin d'éviter qu'une variable à grande échelle numérique domine l'apprentissage du modèle.

## 3. Autoencodeur

L'autoencodeur est un réseau de neurones non supervisé composé de deux parties :

- un encodeur, qui compresse les variables d'entrée dans un espace latent réduit ;
- un décodeur, qui tente de reconstruire les données initiales à partir de cette représentation latente.

L'objectif d'entraînement consiste à minimiser l'erreur de reconstruction entre les données originales et les données reconstruites.

## 4. Représentation latente

L'encodeur apprend une représentation compacte des profils témoins. Cette représentation latente capture les structures dominantes, les corrélations et les variations internes du groupe étudié.

## 5. SHAP

SHAP est appliqué sur l'encodeur afin d'expliquer quelles variables contribuent le plus à la formation de la représentation latente. Dans ce contexte, SHAP ne prédit pas la maladie. Il explique la fonction apprise par l'encodeur.

## 6. Combinaisons de facteurs

Les scores SHAP sont ensuite utilisés pour calculer un impact moyen des combinaisons de deux facteurs. Les combinaisons ayant les valeurs les plus élevées sont considérées comme les plus influentes dans la structuration de l'espace latent des témoins.
