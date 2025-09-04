# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from io import BytesIO

# -------------------------------
# Configuración de página
# -------------------------------
st.set_page_config(page_title="Modelo de Proyección de Ventas", layout="wide")
st.title("📊 Proyección de Ventas con Variables Macroeconómicas")

# -------------------------------
# Cargar datos históricos
# -------------------------------
st.subheader("📂 Cargar Datos Históricos")
st.markdown("El archivo debe tener estas columnas: **Año, PIB, Desempleo, TipoCambioPct, Inflacion, Ventas**")

uploaded_file = st.file_uploader("Carga un archivo CSV o Excel con los datos históricos", type=["csv", "xlsx"])

if uploaded_file:
    # Leer archivo
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Normalizar nombres de columnas (quita acentos y espacios)
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "")
        .str.replace("á", "a")
        .str.replace("í", "i")
        .str.replace("ó", "o")
        .str.replace("ú", "u")
        .str.replace("é", "e")
    )

    st.write("🔎 Columnas detectadas:", df.columns.tolist())
    st.dataframe(df, use_container_width=True)

    # -------------------------------
    # Calcular correlaciones y regresiones simples
    # -------------------------------
    try:
        X_vars = ["PIB", "Desempleo", "TipoCambioPct", "Inflacion"]
        y = df["Ventas"].values

        pendientes, intersecciones, r2_scores = {}, {}, {}

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
        # Pronósticos
        # -------------------------------
        st.subheader("🔮 Pronósticos Macroeconómicos")

        option = st.radio("¿Cómo quieres cargar los pronósticos?", ["Ingresar manualmente", "Subir archivo"])

        forecast = {}
        if option == "Ingresar manualmente":
            col1, col2, col3, col4 = st.columns(4)
            forecast["PIB"] = col1.number_input("Variación PIB (%)", value=2.50)
            forecast["Desempleo"] = col2.number_input("Desempleo (%)", value=3.90)
            forecast["TipoCambioPct"] = col3.number_input("Tipo de Cambio %", value=0.28)
            forecast["Inflacion"] = col4.number_input("Inflación (%)", value=4.80)

        else:
            file_forecast = st.file_uploader("Carga un archivo CSV/Excel con pronósticos", type=["csv", "xlsx"], key="forecast")
            if file_forecast:
                if file_forecast.name.endswith("csv"):
                    df_forecast = pd.read_csv(file_forecast)
                else:
                    df_forecast = pd.read_excel(file_forecast)

                df_forecast.columns = df_forecast.columns.str.strip().str.replace(" ", "")
                st.dataframe(df_forecast, use_container_width=True)
                forecast = df_forecast.iloc[0].to_dict()

        # -------------------------------
        # Cálculo de pronósticos
        # -------------------------------
        if forecast:
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

            # -------------------------------
            # Descargar resultados
            # -------------------------------
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df.to_excel(writer, sheet_name="Datos_Historicos", index=False)
                resultados.to_excel(writer, sheet_name="Regresiones", index=False)
                df_pred.to_excel(writer, sheet_name="Pronosticos", index=False)

            st.download_button(
                label="📥 Descargar resultados en Excel",
                data=output.getvalue(),
                file_name="Resultados_Proyecciones.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    except KeyError:
        st.error("⚠️ Tu archivo debe contener una columna llamada **Ventas** (exactamente). Verifica el nombre en tu dataset.")

else:
    st.warning("⚠️ Sube primero un archivo con los datos históricos.")

