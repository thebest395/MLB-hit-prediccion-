
import joblib
import pandas as pd

# Cargar el modelo
model = joblib.load("mlb_hit_predictor_model.pkl")

# Simulación de datos nuevos
new_data = pd.DataFrame([{
    'is_home': 1,
    'game_month': 7,
    'total_bases': 0,
    'pitcher_throws_R': 1,
    'stand_R': 1,
    'home_team_ATL': 0,
    'home_team_CLE': 0,
    'home_team_HOU': 0,
    'home_team_NYY': 1
}])

# Predecir
prediction = model.predict(new_data)
print("¿Dará hit?:", "Sí" if prediction[0] == 1 else "No")
