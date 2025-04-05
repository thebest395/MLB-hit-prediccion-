
import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

st.set_page_config(page_title="MLB Hit Predictor", layout="centered")

st.title("MLB Hit Predictor (Runtime FIXED)")
st.write("Entrenamiento y predicción en vivo con estructura de features consistente.")

# Inputs
is_home = st.selectbox("¿Juega en casa?", ["Sí", "No"]) == "Sí"
month = st.selectbox("Mes del juego", list(range(4, 10)))
pitcher_r = st.selectbox("Pitcher lanza con...", ["Derecha", "Izquierda"]) == "Derecha"
stand_r = st.selectbox("Bateador es...", ["Derecho", "Zurdo"]) == "Derecho"
home_team = st.selectbox("Equipo local", ["NYY", "LAD", "ATL", "HOU", "CLE"])
home_team_flags = {team: 1 if home_team == team else 0 for team in ["LAD", "ATL", "CLE", "HOU", "NYY"]}

# Construir input
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

# Crear datos sintéticos con mismas columnas
def generate_training_data(n=500):
    np.random.seed(42)
    data = pd.DataFrame({
        'is_home': np.random.randint(0, 2, size=n),
        'game_month': np.random.randint(4, 10, size=n),
        'total_bases': np.random.randint(0, 5, size=n),
        'pitcher_throws_R': np.random.randint(0, 2, size=n),
        'stand_R': np.random.randint(0, 2, size=n),
        'home_team_ATL': np.random.randint(0, 2, size=n),
        'home_team_CLE': np.random.randint(0, 2, size=n),
        'home_team_HOU': np.random.randint(0, 2, size=n),
        'home_team_NYY': np.random.randint(0, 2, size=n),
    })
    # Generar target: hit o no
    data['hit'] = (
        0.3 * data['total_bases'] +
        0.2 * data['is_home'] +
        0.1 * data['pitcher_throws_R'] +
        0.1 * data['stand_R'] +
        np.random.normal(0, 0.5, size=n)
    ) > 1.5
    return data.drop(columns='hit'), data['hit'].astype(int)

X_train, y_train = generate_training_data()

# Entrenar modelo
model = GradientBoostingClassifier().fit(X_train, y_train)

# Predicción
if st.button("¿Dará un hit?"):
    prediction = model.predict(input_data)[0]
    st.success("¡Sí dará un hit!") if prediction == 1 else st.warning("No dará un hit.")
