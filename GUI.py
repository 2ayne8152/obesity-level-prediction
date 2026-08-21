"""
Streamlit GUI — Obesity Level Prediction
==========================================
Loads the XGBoost pipeline trained in `xgboost_obesity_best_params.py`
(`csv/xgboost_obesity_model.pkl` + `csv/target_label_encoder.pkl`) and lets
the user enter eating-habit / physical-condition features through sliders
(numeric) and dropdowns (categorical) to get a predicted obesity level.

Run with:
    streamlit run obesity_prediction_app.py

Expects the following files to exist relative to where you launch the app
(same paths the training script saves to):
    csv/xgboost_obesity_model.pkl
    csv/target_label_encoder.pkl
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "pkl/xgboost_obesity_model.pkl"
ENCODER_PATH = "pkl/target_label_encoder.pkl"

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Obesity Level Predictor",
    page_icon="⚖️",
    layout="centered",
)

# --------------------------------------------------------------------------
# LOAD MODEL (cached so it only loads once per session)
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    target_encoder = joblib.load(ENCODER_PATH)
    return model, target_encoder


try:
    model, target_encoder = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = e

# --------------------------------------------------------------------------
# FEATURE DEFINITIONS
# (ranges / defaults taken from the training dataset's summary statistics)
# --------------------------------------------------------------------------
NUMERIC_FEATURES = {
    # name:        (min,   max,   default, step, label,                     unit)
    "Age":    (14.0, 61.0, 24.0,  1.0, "Age", "years"),
    "Height": (1.45, 1.98, 1.70,  0.01, "Height", "m"),
    "Weight": (39.0, 173.0, 86.0, 0.5, "Weight", "kg"),
    "FCVC":   (1.0, 3.0, 2.0,     0.1, "Frequency of vegetable consumption", "1 (never) – 3 (always)"),
    "NCP":    (1.0, 4.0, 3.0,     0.1, "Number of main meals per day", "meals"),
    "CH2O":   (1.0, 3.0, 2.0,     0.1, "Daily water intake", "1 (<1L) – 3 (>2L)"),
    "FAF":    (0.0, 3.0, 1.0,     0.1, "Physical activity frequency", "days/week (scaled)"),
    "TUE":    (0.0, 2.0, 0.5,     0.1, "Time using technology devices", "0 (low) – 2 (high)"),
}

CATEGORICAL_FEATURES = {
    "Gender": ["Female", "Male"],
    "family_history_with_overweight": ["yes", "no"],
    "FAVC": ["no", "yes"],
    "CAEC": ["Sometimes", "Frequently", "Always", "no"],
    "SMOKE": ["no", "yes"],
    "SCC": ["no", "yes"],
    "CALC": ["no", "Sometimes", "Frequently", "Always"],
    "MTRANS": ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"],
}

CATEGORICAL_LABELS = {
    "Gender": "Gender",
    "family_history_with_overweight": "Family history of overweight",
    "FAVC": "Frequent consumption of high-caloric food (FAVC)",
    "CAEC": "Eating between meals (CAEC)",
    "SMOKE": "Smokes",
    "SCC": "Monitors calorie consumption (SCC)",
    "CALC": "Alcohol consumption (CALC)",
    "MTRANS": "Main mode of transportation",
}

# Column order the model's preprocessor was trained on
FEATURE_ORDER = [
    "Gender", "Age", "Height", "Weight", "family_history_with_overweight",
    "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE",
    "CALC", "MTRANS",
]

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.title("⚖️ Obesity Level Predictor")
st.write(
    "Enter eating-habit and physical-condition information below, then "
    "click **Predict** to estimate the obesity level using a tuned "
    "XGBoost model."
)

if not model_loaded:
    st.error(
        f"Could not load the trained model/encoder.\n\n"
        f"Expected files:\n- `{MODEL_PATH}`\n- `{ENCODER_PATH}`\n\n"
        f"Error: {load_error}"
    )
    st.stop()

# --------------------------------------------------------------------------
# INPUT FORM
# --------------------------------------------------------------------------
with st.form("prediction_form"):

    st.subheader("Personal & Physical Information")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox(CATEGORICAL_LABELS["Gender"], CATEGORICAL_FEATURES["Gender"])
        age = st.slider(
            f"{NUMERIC_FEATURES['Age'][4]} ({NUMERIC_FEATURES['Age'][5]})",
            min_value=NUMERIC_FEATURES["Age"][0],
            max_value=NUMERIC_FEATURES["Age"][1],
            value=NUMERIC_FEATURES["Age"][2],
            step=NUMERIC_FEATURES["Age"][3],
        )
    with col2:
        height = st.slider(
            f"{NUMERIC_FEATURES['Height'][4]} ({NUMERIC_FEATURES['Height'][5]})",
            min_value=NUMERIC_FEATURES["Height"][0],
            max_value=NUMERIC_FEATURES["Height"][1],
            value=NUMERIC_FEATURES["Height"][2],
            step=NUMERIC_FEATURES["Height"][3],
        )
        weight = st.slider(
            f"{NUMERIC_FEATURES['Weight'][4]} ({NUMERIC_FEATURES['Weight'][5]})",
            min_value=NUMERIC_FEATURES["Weight"][0],
            max_value=NUMERIC_FEATURES["Weight"][1],
            value=NUMERIC_FEATURES["Weight"][2],
            step=NUMERIC_FEATURES["Weight"][3],
        )

    family_history = st.selectbox(
        CATEGORICAL_LABELS["family_history_with_overweight"],
        CATEGORICAL_FEATURES["family_history_with_overweight"],
    )

    st.divider()
    st.subheader("Eating Habits")
    favc = st.selectbox(CATEGORICAL_LABELS["FAVC"], CATEGORICAL_FEATURES["FAVC"])
    fcvc = st.slider(
        f"{NUMERIC_FEATURES['FCVC'][4]} ({NUMERIC_FEATURES['FCVC'][5]})",
        min_value=NUMERIC_FEATURES["FCVC"][0],
        max_value=NUMERIC_FEATURES["FCVC"][1],
        value=NUMERIC_FEATURES["FCVC"][2],
        step=NUMERIC_FEATURES["FCVC"][3],
    )
    ncp = st.slider(
        f"{NUMERIC_FEATURES['NCP'][4]} ({NUMERIC_FEATURES['NCP'][5]})",
        min_value=NUMERIC_FEATURES["NCP"][0],
        max_value=NUMERIC_FEATURES["NCP"][1],
        value=NUMERIC_FEATURES["NCP"][2],
        step=NUMERIC_FEATURES["NCP"][3],
    )
    caec = st.selectbox(CATEGORICAL_LABELS["CAEC"], CATEGORICAL_FEATURES["CAEC"])
    calc = st.selectbox(CATEGORICAL_LABELS["CALC"], CATEGORICAL_FEATURES["CALC"])
    ch2o = st.slider(
        f"{NUMERIC_FEATURES['CH2O'][4]} ({NUMERIC_FEATURES['CH2O'][5]})",
        min_value=NUMERIC_FEATURES["CH2O"][0],
        max_value=NUMERIC_FEATURES["CH2O"][1],
        value=NUMERIC_FEATURES["CH2O"][2],
        step=NUMERIC_FEATURES["CH2O"][3],
    )

    st.divider()
    st.subheader("Lifestyle")
    smoke = st.selectbox(CATEGORICAL_LABELS["SMOKE"], CATEGORICAL_FEATURES["SMOKE"])
    scc = st.selectbox(CATEGORICAL_LABELS["SCC"], CATEGORICAL_FEATURES["SCC"])
    faf = st.slider(
        f"{NUMERIC_FEATURES['FAF'][4]} ({NUMERIC_FEATURES['FAF'][5]})",
        min_value=NUMERIC_FEATURES["FAF"][0],
        max_value=NUMERIC_FEATURES["FAF"][1],
        value=NUMERIC_FEATURES["FAF"][2],
        step=NUMERIC_FEATURES["FAF"][3],
    )
    tue = st.slider(
        f"{NUMERIC_FEATURES['TUE'][4]} ({NUMERIC_FEATURES['TUE'][5]})",
        min_value=NUMERIC_FEATURES["TUE"][0],
        max_value=NUMERIC_FEATURES["TUE"][1],
        value=NUMERIC_FEATURES["TUE"][2],
        step=NUMERIC_FEATURES["TUE"][3],
    )
    mtrans = st.selectbox(CATEGORICAL_LABELS["MTRANS"], CATEGORICAL_FEATURES["MTRANS"])

    submitted = st.form_submit_button("Predict", use_container_width=True)

# --------------------------------------------------------------------------
# PREDICTION
# --------------------------------------------------------------------------
if submitted:
    input_dict = {
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "family_history_with_overweight": family_history,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans,
    }

    input_df = pd.DataFrame([input_dict])[FEATURE_ORDER]

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    predicted_label = target_encoder.inverse_transform([prediction])[0]

    st.divider()
    st.subheader("Prediction Result")
    st.success(f"**Predicted obesity level:** {predicted_label.replace('_', ' ')}")

    # BMI for context (not used by the model, just informative)
    bmi = weight / (height ** 2)
    st.caption(f"Computed BMI for reference: {bmi:.1f}")

    prob_df = pd.DataFrame(
        {
            "Obesity Level": [c.replace("_", " ") for c in target_encoder.classes_],
            "Probability": probabilities,
        }
    ).sort_values("Probability", ascending=False)

    st.write("Prediction probabilities by class:")
    st.bar_chart(prob_df.set_index("Obesity Level"))
    st.dataframe(prob_df.reset_index(drop=True), use_container_width=True)