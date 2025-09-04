# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# -------------------------------
# Configuración de página
# -------------------------------
st.set_page_config(page_title="Modelo de Proyección de Ventas", layout="wide")
st.title("📊 Proyección de Ventas con Variables Macroeconómicas")

# -------------------------------
# Datos históricos (ejemplo)
# -------------------------------
data = {
    "Año": [2000, 2001, 2002, 2003],
    "PIB": [2.31, 4.50, 2.29, 0.43],
    "Desempleo": [3.50, 4.40, 4.30, 3.80],
    "TipoCambioPct": [-3.47, 0.04, 0.24, 1.93],
    "Inflación": [3.33, 4.05, 3.76, 6.53],
    "Ventas": [44.18, 49.13, 50.29, 33.13],
}
df = pd.DataFrame(data)

st.subheader("📌 Datos Históricos")
st.dataframe(df, use_container_width=True)

# -------------------------------
# Cálculo de correlaciones y regresiones simples
# -------------------------------
X_vars = ["PIB", "Desempleo", "TipoCambioPct", "Inflación"]
y = df["Ventas"].values

pendientes = {}
intersecciones = {}
r2_scores = {}

for var in X_vars:
    X = df[[var]].values
    model = LinearRegression().fit(X, y)
    pendientes[var] = model.coef_[0]
    intersecciones[var] = model.intercept_
    r2_scores[var] = model.score(X, y)

resultados = pd.DataFrame({
    "Variable": X_vars,
    "Pendiente (β)": [pendientes[v] for v in X_vars],
    "Intersección (α)": [intersecciones[v] for v in X_vars],
    "R²": [r2_scores[v] for v in X_vars]
})

st.subheader("📈 Resultados de las Regresiones Simples")
st.dataframe(resultados, use_container_width=True)

# -------------------------------
# Pronósticos de entrada
# -------------------------------
st.subheader("🔮 Pronósticos Macroeconómicos (ejemplo 2004p)")
col1, col2, col3, col4 = st.columns(4)
pib = col1.number_input("Variación PIB (%)", value=2.50)
desempleo = col2.number_input("Desempleo (%)", value=3.90)
tc_pct = col3.number_input("Tipo de Cambio %", value=0.28)
inflacion = col4.number_input("Inflación (%)", value=4.80)

forecast = {"PIB": pib, "Desempleo": desempleo, "TipoCambioPct": tc_pct, "Inflación": inflacion}

# -------------------------------
# Cálculo de pronósticos
# -------------------------------
ventas_pred_simple = {}
for var, x_val in forecast.items():
    y_pred = intersecciones[var] + pendientes[var] * x_val
    ventas_pred_simple[var] = y_pred

# Normalizar los R² como ponderadores
total_r2 = sum(r2_scores.values())
pesos = {var: r2_scores[var]/total_r2 for var in r2_scores}

ventas_pred_ponderada = sum(ventas_pred_simple[var]*pesos[var] for var in forecast)

# -------------------------------
# Mostrar resultados
# -------------------------------
st.subheader("📊 Pronóstico de Ventas con Regresiones Simples")
df_pred = pd.DataFrame({
    "Variable": X_vars,
    "Pronóstico Ventas (%)": [ventas_pred_simple[v] for v in X_vars],
    "Peso (R²)": [pesos[v] for v in X_vars]
})
st.dataframe(df_pred, use_container_width=True)

st.subheader("⚖️ Pronóstico de Ventas con Regresión Múltiple Ponderada")
st.metric(label="Ventas proyectadas (%)", value=f"{ventas_pred_ponderada:.2f}")
