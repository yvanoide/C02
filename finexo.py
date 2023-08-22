import streamlit as st
import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

data_model = pd.read_csv("data_model.csv")
New_Data = data_model[['Carrosserie', 'masse_ordma_min', 'masse_ordma_max', 'co2']]
numeric_columns = ['masse_ordma_min', 'masse_ordma_max']
X = New_Data[numeric_columns]
Y = New_Data['co2']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

models = [
    ('Dummy', DummyRegressor(), {}),
    ('LinearRegression', LinearRegression(), {}),
    ('SGDRegressor', SGDRegressor(), {}),
    ('RandomForestRegressor', RandomForestRegressor(), {}),
    ('GradientBoostingRegressor', GradientBoostingRegressor(), {}),
    ('XGBRegressor', XGBRegressor(), {})
]

scoring = {
    'r2': make_scorer(r2_score),
    'neg_mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False),
    'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False)
}


st.title('Prédiction de CO2 pour un véhicule')


carrosserie_options = sorted(data_model['Carrosserie'].unique())
selected_carrosserie = st.selectbox("Sélectionnez le type de carrosserie :", carrosserie_options)


masse_ordma_min = st.number_input("Entrez la masse ordma_min :", min_value=0)
masse_ordma_max = st.number_input("Entrez la masse ordma_max :", min_value=0)

if st.button("Prédire CO2"):
    
    filtered_data = data_model[
        (data_model['Carrosserie'] == selected_carrosserie) &
        (data_model['masse_ordma_min'] >= masse_ordma_min) &
        (data_model['masse_ordma_max'] <= masse_ordma_max)
    ]

    if not filtered_data.empty:
       
        features = filtered_data[['masse_ordma_min', 'masse_ordma_max']]

      
        results = {}
        for name, model, param_grid in models:
            print(f"Training {name}...")
            grid_search = GridSearchCV(model, param_grid, scoring=scoring, refit='r2', cv=5, verbose=2)
            grid_search.fit(X_train, y_train)
            results[name] = grid_search

        # Trouver le meilleur modèle
        best_model = max(results.keys(), key=lambda name: results[name].best_score_)
        
        # Prédire avec le meilleur modèle
        best_model_predictions = results[best_model].predict(features)
        predicted_co2 = best_model_predictions[0] if isinstance(best_model_predictions, (list, np.ndarray)) else best_model_predictions
        
        # Afficher la prédiction
        st.write("Prédiction de CO2 en grammes par kilomètre avec le meilleur modèle :", round(predicted_co2, 2))
        
    else:
        st.warning("Aucune donnée disponible pour ces critères.")
