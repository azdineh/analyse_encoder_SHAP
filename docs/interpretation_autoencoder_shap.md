# Interprétation Autoencodeur + SHAP

## Question méthodologique

Dans cette analyse, l'autoencodeur est un modèle non supervisé, alors que SHAP est souvent utilisé pour expliquer des modèles supervisés. Cette situation demande une clarification importante.

## Principe

SHAP n'explique pas nécessairement une classe ou une variable cible. SHAP explique la sortie d'une fonction apprise par un modèle. La fonction peut être un classifieur supervisé, un régresseur, ou une composante interne d'un réseau de neurones comme l'encodeur d'un autoencodeur.

## Application dans cette étude

Dans ce travail, l'autoencodeur apprend la transformation suivante :

```text
X -> Encodeur -> Z
```

- `X` représente les variables observées.
- `Z` représente l'espace latent appris.

SHAP est utilisé pour expliquer comment les variables de `X` contribuent à la formation de `Z`.

## Interprétation correcte

Les valeurs SHAP indiquent les variables qui influencent le plus la représentation latente apprise par l'autoencodeur. Elles reflètent donc l'importance structurelle des facteurs dans les profils témoins.

## Ce que les résultats permettent d'affirmer

Les résultats permettent d'identifier les variables qui structurent fortement les données témoins. Ces variables peuvent être considérées comme importantes dans l'organisation interne des profils étudiés.

## Ce que les résultats ne permettent pas d'affirmer

Les résultats ne permettent pas de conclure que ces variables sont des facteurs de risque du cancer. Ils ne démontrent pas une relation causale ou prédictive avec la pathologie, car l'analyse est réalisée sans variable cible et uniquement sur les cas témoins.

## Formulation scientifique proposée

Les résultats SHAP obtenus permettent d'identifier les variables contribuant le plus à la structuration de l'espace latent appris par l'autoencodeur. Ces résultats reflètent l'importance relative des facteurs dans la représentation interne des données témoins, sans établir une relation directe, causale ou prédictive avec la survenue de la pathologie.
