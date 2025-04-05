
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="MLB Hit Predictor", layout="centered")

st.title("MLB Hit Predictor")
st.write("Selecciona las características del turno para predecir si el jugador dará un hit.")

# Cargar el modelo
model = joblib.load("mlb_hit_predictor_model.pkl")

# Inputs
is_home = st.selectbox("¿Juega en casa?", ["Sí", "No"]) == "Sí"
month = st.selectbox("Mes del juego", list(range(4, 10)))
pitcher_r = st.selectbox("Pitcher lanza con...", ["Derecha", "Izquierda"]) == "Derecha"
stand_r = st.selectbox("Bateador es...", ["Derecho", "Zurdo"]) == "Derecho"
home_team = st.selectbox("Equipo local", ["NYY", "LAD", "ATL", "HOU", "CLE"])
home_team_flags = {team: 1 if home_team == team else 0 for team in ["LAD", "ATL", "CLE", "HOU", "NYY"]}

# Construir DataFrame para predicción
input_data = pd.DataFrame([{
    'is_home': int(is_home),
    'game_month': month,
    'total_bases': 0,
    'pitcher_throws_R': int(pitcher_r),
    'stand_R': int(stand_r),
    **{f'home_team_{team}': v for team, v in home_team_flags.items() if team != "LAD"}  # drop_first
}])

# Predicción
if st.button("¿Dará un hit?"):
    pred = model.predict(input_data)[0]
    st.success("¡Sí dará un hit!") if pred == 1 else st.warning("No dará un hit.")
