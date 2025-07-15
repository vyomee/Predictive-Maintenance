import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Predictive Maintenance: RUL Estimator",
    page_icon="🤖",
    layout="wide",
)

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        return joblib.load("random_forest_model.joblib")
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'random_forest_model.joblib' is in the root directory.")
        st.stop()

model = load_model()

# --- Helper Function ---
def calculate_rul(failure_proba):
    return 1 - failure_proba

# --- Data Ranges and Features ---
data_ranges = {
    'age': (0, 20), 'day_cos': (-1, 1), 'day_of_month': (1, 31), 'day_of_week': (0, 6), 'day_sin': (-1, 1),
    'hour': (0, 23), 'hour_cos': (-1, 1), 'hour_sin': (-1, 1), 'month': (1, 12), 'model': (0, 3),
    'pressure': (90, 110), 'pressure_lag_12h': (90, 110), 'pressure_lag_1h': (90, 110),
    'pressure_lag_3h': (90, 110), 'pressure_lag_6h': (90, 110), 'pressure_rolling_max_12h': (100, 130),
    'pressure_rolling_max_24h': (100, 140), 'pressure_rolling_max_6h': (100, 120), 'pressure_rolling_mean_12h': (90, 110),
    'pressure_rolling_mean_24h': (90, 110), 'pressure_rolling_mean_6h': (90, 110), 'pressure_rolling_min_12h': (70, 90),
    'pressure_rolling_min_24h': (60, 90), 'pressure_rolling_min_6h': (80, 95), 'pressure_rolling_std_12h': (1, 15),
    'pressure_rolling_std_24h': (1, 20), 'pressure_rolling_std_6h': (1, 10), 'quarter': (1, 4), 'rotate': (400, 500),
    'rotate_lag_12h': (400, 500), 'rotate_lag_1h': (400, 500), 'rotate_lag_3h': (400, 500),
    'rotate_lag_6h': (400, 500), 'rotate_rolling_max_12h': (450, 550), 'rotate_rolling_max_24h': (450, 600),
    'rotate_rolling_max_6h': (450, 520), 'rotate_rolling_mean_12h': (400, 500), 'rotate_rolling_mean_24h': (400, 500),
    'rotate_rolling_mean_6h': (400, 500), 'rotate_rolling_min_12h': (350, 450), 'rotate_rolling_min_24h': (300, 450),
    'rotate_rolling_min_6h': (380, 450), 'rotate_rolling_std_12h': (10, 50), 'rotate_rolling_std_24h': (10, 60),
    'rotate_rolling_std_6h': (10, 40), 'vibration': (35, 50), 'vibration_lag_12h': (35, 50),
    'vibration_lag_1h': (35, 50), 'vibration_lag_3h': (35, 50), 'vibration_lag_6h': (35, 50),
    'vibration_rolling_max_12h': (40, 60), 'vibration_rolling_max_24h': (40, 70), 'vibration_rolling_max_6h': (40, 55),
    'vibration_rolling_mean_12h': (35, 50), 'vibration_rolling_mean_24h': (35, 50), 'vibration_rolling_mean_6h': (35, 50),
    'vibration_rolling_min_12h': (25, 40), 'vibration_rolling_min_24h': (20, 40), 'vibration_rolling_min_6h': (30, 40),
    'vibration_rolling_std_12h': (1, 10), 'vibration_rolling_std_24h': (1, 12), 'vibration_rolling_std_6h': (1, 8),
    'volt': (160, 180), 'volt_lag_12h': (160, 180), 'volt_lag_1h': (160, 180), 'volt_lag_3h': (160, 180),
    'volt_lag_6h': (160, 180), 'volt_rolling_max_12h': (170, 190), 'volt_rolling_max_24h': (170, 200),
    'volt_rolling_max_6h': (170, 185), 'volt_rolling_mean_12h': (160, 180), 'volt_rolling_mean_24h': (160, 180),
    'volt_rolling_mean_6h': (160, 180), 'volt_rolling_min_12h': (140, 170), 'volt_rolling_min_24h': (130, 170),
    'volt_rolling_min_6h': (150, 170), 'volt_rolling_std_12h': (5, 15), 'volt_rolling_std_24h': (5, 20),
    'volt_rolling_std_6h': (5, 10)
}

feature_columns = list(data_ranges.keys())

# --- UI Content ---
st.title("⚙️ Predictive Maintenance: Failure Predictor")

st.markdown("""
This application uses a pre-trained **Random Forest Classifier** to predict the likelihood of a machine component failure based on simulated sensor data.

**How it Works:**
1. Set the machine's expected lifespan in the sidebar.
2. Click **Generate & Predict**.
3. Random sensor values are generated.
4. Failure probability and Remaining Useful Life (RUL) are displayed.
---
""")

with st.sidebar:
    st.header("Project Details")
    st.markdown("""
    ### Model Information
    - **Type:** Random Forest Classifier
    - **Target:** Imminent Failure (Yes/No)
    - **Features:** 79 engineered sensor & time-based features

    ### Data Source
    - [Azure Predictive Maintenance Dataset](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)

    ### Feature Engineering
    - Rolling stats, lags, cyclical time encoding
    """)

    st.divider()
    st.header("Machine Lifespan")
    lifespan_value = st.number_input("Expected lifespan:", min_value=1.0, value=20000.0, step=1000.0)
    lifespan_unit = st.selectbox("Time unit:", ["Hours", "Months", "Years"])

st.header("Live Failure Prediction")
col1, col2 = st.columns([1, 2])

with col1:
    if st.button("Generate & Predict", type="primary", use_container_width=True):
        rand_values = {}
        for feat in feature_columns:
            if feat not in data_ranges:
                st.warning(f"Feature '{feat}' missing from data_ranges. Using 0.")
                rand_values[feat] = 0
                continue
            low, high = data_ranges[feat]
            if feat in ['model', 'day_of_month', 'day_of_week', 'hour', 'month', 'quarter', 'age']:
                rand_values[feat] = np.random.randint(low, high + 1)
            else:
                rand_values[feat] = np.random.uniform(low, high)

        rand_data = pd.DataFrame([rand_values], columns=feature_columns)
        pred_proba = model.predict_proba(rand_data)[0]
        failure_probability = pred_proba[1]
        rul_percentage = calculate_rul(failure_probability)

        st.session_state['rand_data'] = rand_data
        st.session_state['rul_percentage'] = rul_percentage
        st.session_state['failure_probability'] = failure_probability

with col2:
    if 'rul_percentage' in st.session_state:
        rul_percentage = st.session_state['rul_percentage']
        rand_data = st.session_state['rand_data']
        failure_probability = st.session_state['failure_probability']

        st.write("---")
        st.write("#### Calculate RUL from Expected Lifespan")
        col3, col4 = st.columns(2)
        with col3:
            lifespan = st.number_input("Enter expected lifespan:", min_value=1, step=1)
        with col4:
            unit = st.selectbox("Select unit:", ["Hours", "Months", "Years"])

        if st.button("Calculate RUL from Lifespan", use_container_width=True):
            estimated_rul = lifespan * rul_percentage
            st.success(f"Estimated RUL: {estimated_rul:.2f} {unit.lower()}")

        st.write("#### Prediction Result: Remaining Useful Life (RUL)")
        st.subheader("Prediction Results")

        if rul_percentage < 0.3:
            st.error(f"**Status:** Critical ({rul_percentage*100:.1f}% RUL)", icon="🚨")
        elif rul_percentage < 0.7:
            st.warning(f"**Status:** Approaching Maintenance ({rul_percentage*100:.1f}% RUL)", icon="⚠️")
        else:
            st.success(f"**Status:** Operational ({rul_percentage*100:.1f}% RUL)", icon="✅")

        st.divider()

        colA, colB = st.columns(2)
        with colA:
            st.metric("Failure Probability", f"{failure_probability*100:.1f}%")
        with colB:
            st.metric(f"Estimated RUL in {lifespan_unit}", f"{lifespan_value * rul_percentage:.1f}")

        with st.expander("View Generated Sensor Data"):
            st.dataframe(rand_data)
    else:
        st.info("Click 'Generate & Predict' to begin.")
