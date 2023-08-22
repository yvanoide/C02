# C02
Voici un projet d'etudes de machine learning dans le thème des émissions de gaz à effet de serre au sein des véhicules.

# Projet de Prédiction de CO2 pour un Véhicule

Ce projet consiste en une application Streamlit permettant de prédire le niveau d'émission de CO2 pour un véhicule en fonction de différents paramètres tels que le type de carrosserie et la masse.

## Prérequis

Assurez-vous d'avoir les bibliothèques Python suivantes installées:

- pandas
- scikit-learn
- xgboost
- streamlit


Le projet est organisé de la manière suivante:

- `data_model.csv`: Fichier de données contenant les informations sur le type de carrosserie, la masse et le niveau d'émission de CO2.
- `finexo.py`: Le code source de l'application Streamlit pour la prédiction de CO2.

## Utilisation

1. Assurez-vous d'avoir installé les bibliothèques requises.
2. Placez le fichier `data_model.csv` dans le même répertoire que `finexo.py`.
3. Exécutez l'application Streamlit



4. L'application se lancera dans votre navigateur. Vous pourrez sélectionner le type de carrosserie, entrer les valeurs de masse ordma_min et ordma_max, puis cliquer sur le bouton "Prédire CO2" pour obtenir les prédictions de CO2 en fonction des modèles sélectionnés.

## Modèles de Régression

Le projet utilise plusieurs modèles de régression pour prédire le niveau d'émission de CO2:

- DummyRegressor
- LinearRegression
- SGDRegressor
- RandomForestRegressor
- GradientBoostingRegressor
- XGBRegressor

## Optimisation des Hyperparamètres

Les modèles ont été optimisés en utilisant une recherche par grille (GridSearch) pour trouver les meilleurs hyperparamètres. Les scores R2, MAE et RMSE ont été utilisés pour l'évaluation.

Yves
