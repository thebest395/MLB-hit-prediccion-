
import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification

st.set_page_config(page_title="MLB Hit Predictor", layout="centered")

st.title("MLB Hit Predictor (Runtime)")
st.write("Simulación: predicción en vivo basada en datos sintéticos y entrenamiento en tiempo real.")

# Simular datos de entrenamiento
X, y = make_classification(n_samples=500, n_features=7, random_state=42)
model = GradientBoostingClassifier().fit(X, y)

# Inputs
is_home = st.selectbox("¿Juega en casa?", ["Sí", "No"]) == "Sí"
month = st.selectbox("Mes del juego", list(range(4, 10)))
pitcher_r = st.selectbox("Pitcher lanza con...", ["Derecha", "Izquierda"]) == "Derecha"
stand_r = st.selectbox("Bateador es...", ["Derecho", "Zurdo"]) == "Derecho"
home_team = st.selectbox("Equipo local", ["NYY", "LAD", "ATL", "HOU", "CLE"])
home_team_flags = {team: 1 if home_team == team else 0 for team in ["LAD", "ATL", "CLE", "HOU", "NYY"]}

# Armar datos para predecir
input_data = pd.DataFrame([{
    'is_home': int(is_home),
    'game_month': month,
    'total_bases': 0,
    'pitcher_throws_R': int(pitcher_r),
    'stand_R': int(stand_r),
    'home_team_ATL': home_team_flags['ATL'],
    'home_team_CLE': home_team_flags['CLE'],
    'home_team_HOU': home_team_flags['HOU'],
    'home_team_NYY': home_team_flags['NYY']
}])

if st.button("¿Dará un hit?"):
    prediction = model.predict(input_data)[0]
    st.success("¡Sí dará un hit!") if prediction == 1 else st.warning("No dará un hit.")
