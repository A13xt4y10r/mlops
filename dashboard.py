import streamlit as st
import pandas as pd
import joblib
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from utils import create_new_features, drop_unwanted_rows, drop_outlayers
from constants import TARGET_COLUMN_NAME, BASE_COLUMN_NAME, TARGET_COLUMN_ORIGINAL_NAME
from MLModel import MLModel

# Betanított modell betöltése
model = joblib.load('artifacts/models/randomforest_model.pkl')

# Adatok betöltése
def load_data(file):
    try:
        return pd.read_csv(file)
    except pd.errors.ParserError as e:
        st.error(f"Hiba történt a CSV fájl beolvasása közben: {e}")
        return None

# Predikciók készítése
def make_predictions(data):
    return model.predict(data)

# EvidentlyAI Data Drift jelentés generálása
def generate_data_drift_report(reference_data, current_data):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    return report

# Streamlit alkalmazás
st.title('Számla Kifizetési Idő Predikció Dashboard')

uploaded_file = st.file_uploader("Válassz egy CSV fájlt", type="csv")
reference_file = st.file_uploader("Válassz egy referencia CSV fájlt", type="csv")

if uploaded_file:
    data = load_data(uploaded_file)
    if data is not None:
        st.write("Adatok előnézete:")
        st.write(data.head())

        data = drop_unwanted_rows(data)

        # Konvertáld az oszlopokat dátum típusúra
        data[BASE_COLUMN_NAME] = pd.to_datetime(data[BASE_COLUMN_NAME])
        data[TARGET_COLUMN_ORIGINAL_NAME] = pd.to_datetime(data[TARGET_COLUMN_ORIGINAL_NAME])

        # Adatok előfeldolgozása a predikcióhoz
        ml_model = MLModel()
        data = ml_model.preprocessing_pipeline_inference(data)

        if st.button('Predikciók készítése'):
            predictions = make_predictions(data)
            data['Predicted Payment Time'] = predictions
            st.write("Predikciók:")
            st.write(data)

            st.bar_chart(data['Predicted Payment Time'])

            csv = data.to_csv(index=False)
            st.download_button(label="Predikciók letöltése CSV fájlban", data=csv, file_name="predictions.csv", mime="text/csv")

if reference_file:
    reference_data = load_data(reference_file)
    if reference_data is not None:
        reference_data = drop_unwanted_rows(reference_data)

        # Konvertáld az oszlopokat dátum típusúra
        reference_data[BASE_COLUMN_NAME] = pd.to_datetime(reference_data[BASE_COLUMN_NAME])
        reference_data[TARGET_COLUMN_ORIGINAL_NAME] = pd.to_datetime(reference_data[TARGET_COLUMN_ORIGINAL_NAME])

        # Adatok előfeldolgozása a predikcióhoz
        reference_data = ml_model.preprocessing_pipeline_inference(reference_data)

        if st.button('Data Drift jelentés generálása'):
            report = generate_data_drift_report(reference_data, data)
            report_path = "artifacts/data_drift_report.html"
            report.save_html(report_path)
            st.success(f"Data Drift jelentés mentve: {report_path}")

            with open(report_path, "rb") as file:
                btn = st.download_button(
                    label="Data Drift jelentés letöltése",
                    data=file,
                    file_name="data_drift_report.html",
                    mime="text/html"
                )

            with open(report_path, "r") as f:
                html = f.read()
            st.components.v1.html(html, height=800, scrolling=True)
else:
    st.write("Kérlek, tölts fel egy referencia CSV fájlt az adatok drift ellenőrzéséhez.")