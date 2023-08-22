import streamlit as st
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV

# Charger et prétraiter les données
data_model = pd.read_csv("data_model.csv")
New_Data = data_model[['Carrosserie', 'masse_ordma_min', 'masse_ordma_max', 'co2']]
numeric_columns = ['masse_ordma_min', 'masse_ordma_max']
X = New_Data[numeric_columns]
Y = New_Data['co2']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Définir les modèles et les hyperparamètres à rechercher
models = [
    ('DummyRegressor', DummyRegressor(), {}),
    ('LinearRegression', LinearRegression(), {}),
    ('SGDRegressor', SGDRegressor(), {'max_iter': [100, 500]}),
    ('RandomForestRegressor', RandomForestRegressor(), {'n_estimators': [100, 200, 300]}),
    ('GradientBoostingRegressor', GradientBoostingRegressor(), {'n_estimators': [100, 200, 300]}),
    ('XGBRegressor', XGBRegressor(), {'n_estimators': [100, 200, 300]})
]

scoring = {
    'r2': 'r2',
    'neg_mean_absolute_error': 'neg_mean_absolute_error',
    'neg_mean_squared_error': 'neg_mean_squared_error'
}

# Interface utilisateur Streamlit
st.title('Prédiction de CO2 pour un véhicule')

# Sélection du type de carrosserie
carrosserie_options = sorted(data_model['Carrosserie'].unique())
selected_carrosserie = st.selectbox("Sélectionnez le type de carrosserie :", carrosserie_options)

# Saisie de la masse ordma_min et ordma_max
masse_ordma_min = st.number_input("Entrez la masse ordma_min :", min_value=0)
masse_ordma_max = st.number_input("Entrez la masse ordma_max :", min_value=0)

# Bouton pour lancer la prédiction
if st.button("Prédire CO2"):
    # Filtrer les données en fonction des choix de l'utilisateur
    filtered_data = data_model[
        (data_model['Carrosserie'] == selected_carrosserie) &
        (data_model['masse_ordma_min'] >= masse_ordma_min) &
        (data_model['masse_ordma_max'] <= masse_ordma_max)
    ]

    if not filtered_data.empty:
        st.write("Comparaison des performances des modèles :")

        for name, model, param_grid in models:
            print(f"Training {name}...")
            grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=5, verbose=2)
            grid_search.fit(X_train, y_train)

            # Utiliser le modèle ajusté pour les prédictions
            predictions = grid_search.predict(X_test)
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = mean_squared_error(y_test, predictions, squared=False)

            st.write(f"Modèle : {name}")
            st.write(f"Score R2 : {r2:.4f}")
            st.write(f"Mean Absolute Error (MAE) : {mae:.4f}")
            st.write(f"Root Mean Squared Error (RMSE) : {rmse:.4f}")
            st.write("-------------------")
    else:
        st.warning("Aucune donnée disponible pour ces critères.")
