import streamlit as st
import pandas as pd
import numpy as np
import glob
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Any
import time
import tensorflow as tf

models_dir = "./out"
DEBOUNCE_SECONDS = 1.0
if 'force_run_predict' not in st.session_state:
    st.session_state['force_run_predict'] = False

# SUPER FUNCTIONS
def load_models(models_dir: str = models_dir) -> Dict[str, Any]:
    models = {}
    path = Path(models_dir)
    if not path.exists():
        return models
    for file in path.glob("*Model.pkl"):
        model = pickle.load(open(file, "rb"))
        models[file.stem.removesuffix("Model")] = model # stem => no extension
    for file in path.glob("*Model.keras"):
        model = tf.keras.models.load_model(file)
        models[file.stem.removesuffix("Model")] = model # stem => no extension
    return models

def load_metrics(metrics_path: str = models_dir) -> Dict[str, Dict[str, float]]:
    metrics_dict = {}
    path = Path(metrics_path)
    if not path.exists():
        return metrics_dict
    for file in path.glob("*Meta.json"):
        metrics = json.load(open(file, "r"))
        metrics_dict[file.stem.removesuffix("Meta")] = metrics # stem => no extension
    return metrics_dict

def load_scalers(scalers_path: str = models_dir) -> Dict[str, Any]:
    scalers = {}
    path = Path(scalers_path)
    if not path.exists():
        return scalers
    for file in path.glob("scaler*.pkl"):
        scaler = pickle.load(open(file, "rb"))
        scalers[file.stem] = scaler # stem => no extension
    return scalers

def mark_change():
    st.session_state['force_run_predict'] = True

# -----------------------------------------------------------------------------
st.set_page_config(page_title="California households — regression UI", layout="wide")

st.sidebar.header("Входные характеристики (features)")
median_house_value = st.sidebar.number_input(
    "Median House Value - медианная цена дома в квартале [$]",
    min_value=0.0, value=100000.0, step=1000.0, format="%.2f",
    on_change=mark_change
)
median_age = st.sidebar.number_input(
    "Median Age - медианный возраст домов в квартале [лет] (меньше = новее)",
    min_value=0.0, max_value=200.0, value=20.0, step=1.0,
    on_change=mark_change
)
total_rooms = st.sidebar.slider(
    "Total Rooms - общее количество комнат в квартале",
    min_value=1, max_value=10000, value=2000, step=1,
    on_change=mark_change
)
population = st.sidebar.number_input(
    "Population - общее количество людей в квартале",
    min_value=1, max_value=1000000, value=1500, step=1,
    on_change=mark_change
)

# kinda hard
if 'dist_coast' not in st.session_state:
    st.session_state['dist_coast'] = 10000.0

st.sidebar.write("Distance to coast - расстояние до побережья [м]")
col1, col2 = st.sidebar.columns([2,1])
with col1:
    dist_slider = st.slider(
        "slider",
        min_value=0,
        max_value=200000,
        value=int(st.session_state['dist_coast']),
        step=100,
        on_change=mark_change
    )
with col2:
    dist_input = st.number_input(
        "[м]",
        min_value=0.0,
        max_value=3e6,
        value=float(st.session_state['dist_coast']),
        step=100.0,
        format="%.1f",
        on_change=mark_change
    )
if dist_slider != int(st.session_state['dist_coast']):
    st.session_state['dist_coast'] = float(dist_slider)
elif dist_input != st.session_state['dist_coast']:
    st.session_state['dist_coast'] = float(dist_input)
distance_to_coast = st.session_state['dist_coast']

st.sidebar.markdown("---")
save_predict = st.sidebar.button("Наколдовать магию и сохранить!", on_click=mark_change)

# -----------------------------------------------------------------------------
models = load_models()
metrics = load_metrics()
scalers = load_scalers()
if len(models) == 0:
    st.sidebar.error(f"Модели не найдены в {models_dir} (*.pkl). А ну побежал тренить!")

# -----------------------------------------------------------------------------
st.title("Прогноз медианного дохода (Median Income) — California households")
st.markdown(
    "Вводите значения признаков в боковой панели. Для каждой модели будет показано предсказание и метрики"
)

# current input values
input_df = pd.DataFrame({
    "Median_House_Value": [median_house_value],
    "Distance_to_coast": [distance_to_coast],
    "Tot_Rooms": [total_rooms],
    "Median_Age": [median_age],
    "Population": [population]
}) # dataVector to df

st.subheader("Текущие входные значения")
st.dataframe(input_df)

# -----------------------------------------------------------------------------
if 'last_pred_time' not in st.session_state:
    st.session_state['last_pred_time'] = 0.0
    
now = time.time()
time_since_change = now - st.session_state['last_pred_time']

results = []
# for each model make prediction and show metrics
if (time_since_change >= DEBOUNCE_SECONDS or st.session_state['force_run_predict'] == True) and save_predict:
    for model_name in models.keys():
        model = models.get(model_name)
        if model is None:
            st.error(f"Модель {model_name} не загружена??!!?")
            continue
        
        mustScale = metrics.get(model_name).get('mustScale')
        pred = 0
        if (mustScale):
            scalerX = scalers.get("scalerNormForX")
            inputScaled = scalerX.transform(input_df)
            predScaled = model.predict(inputScaled)
            scalerY = scalers.get("scalerNormForY")
            pred = scalerY.inverse_transform(predScaled)[0]
        else:
            pred = model.predict(input_df)
        
        model_metrics = metrics.get(model_name)
        r2 = model_metrics.get('R2')
        rmse = model_metrics.get('RMSE')
        mae = model_metrics.get('MAE')

        results.append({
            'model': model_name,
            'prediction': pred[0],
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        })
    st.session_state['last_pred_time'] = now
    force_run_predict = False

# results
st.subheader("Результаты колдовства")
cols = st.columns(max(1, len(results)))
for column, result in zip(cols, results):
    with column:
        st.metric(label=f"Модель: {result['model']}",
                  value=f"{result['prediction']:.4f}"
        )
        st.write(f"**r2:** {result['r2']:.4f}")
        st.write(f"**RMSE:** {result['rmse']:.4f}")
        st.write(f"**MAE:** {result['mae']:.4f}")

# run magic action + logs 'cause we're cool kids SWAG
if save_predict:
    if 'pred_log' not in st.session_state:
        st.session_state['pred_log'] = []
    for result in results:
        st.session_state['pred_log'].append({
            'Model': result['model'],
            'Prediction': result['prediction'],
            "Median House Value": input_df["Median_House_Value"][0],
            "Distance to coast": input_df["Distance_to_coast"][0],
            "Total Rooms": input_df["Tot_Rooms"][0],
            "Median Age": input_df["Median_Age"][0],
            "Population": input_df["Population"][0]
        })

if 'pred_log' in st.session_state and len(st.session_state['pred_log']) > 0:
    st.subheader('История магии')
    log_df = pd.DataFrame(st.session_state['pred_log'])
    st.dataframe(log_df.tail(20))