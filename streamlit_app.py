import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading the pre-trained models

# --- Page Configuration ---
st.set_page_config(
    page_title="Predictive Maintenance: RUL Estimator",
    page_icon="🤖",
    layout="wide"
)

# --- Model, Feature Loading, and Helper Functions ---
try:
    # Load the pre-trained Random Forest model
    model = joblib.load("random_forest_model.joblib")
except FileNotFoundError:
        st.error("Model file not found. Please ensure 'random_forest_model.joblib' is in the root directory.")
        st.stop()

# This is the full list of 79 features the model was trained on
feature_columns = [
    'age', 'day_cos', 'day_of_month', 'day_of_week', 'day_sin', 'hour', 'hour_cos', 'hour_sin', 'month', 'model',
    'pressure', 'pressure_lag_12h', 'pressure_lag_1h', 'pressure_lag_3h', 'pressure_lag_6h',
    'pressure_rolling_max_12h', 'pressure_rolling_max_24h', 'pressure_rolling_max_6h', 'pressure_rolling_mean_12h',
    'pressure_rolling_mean_24h', 'pressure_rolling_mean_6h', 'pressure_rolling_min_12h', 'pressure_rolling_min_24h',
    'pressure_rolling_min_6h', 'pressure_rolling_std_12h', 'pressure_rolling_std_24h', 'pressure_rolling_std_6h',
    'quarter', 'rotate', 'rotate_lag_12h', 'rotate_lag_1h', 'rotate_lag_3h', 'rotate_lag_6h',
    'rotate_rolling_max_12h', 'rotate_rolling_max_24h', 'rotate_rolling_max_6h', 'rotate_rolling_mean_12h',
    'rotate_rolling_mean_24h', 'rotate_rolling_mean_6h', 'rotate_rolling_min_12h', 'rotate_rolling_min_24h',
    'rotate_rolling_min_6h', 'rotate_rolling_std_12h', 'rotate_rolling_std_24h', 'rotate_rolling_std_6h',
    'vibration', 'vibration_lag_12h', 'vibration_lag_1h', 'vibration_lag_3h', 'vibration_lag_6h',
    'vibration_rolling_max_12h', 'vibration_rolling_max_24h', 'vibration_rolling_max_6h',
    'vibration_rolling_mean_12h', 'vibration_rolling_mean_24h', 'vibration_rolling_mean_6h',
    'vibration_rolling_min_12h', 'vibration_rolling_min_24h', 'vibration_rolling_min_6h',
    'vibration_rolling_std_12h', 'vibration_rolling_std_24h', 'vibration_rolling_std_6h', 'volt',
    'volt_lag_12h', 'volt_lag_1h', 'volt_lag_3h', 'volt_lag_6h', 'volt_rolling_max_12h', 'volt_rolling_max_24h',
    'volt_rolling_max_6h', 'volt_rolling_mean_12h', 'volt_rolling_mean_24h', 'volt_rolling_mean_6h',
    'volt_rolling_min_12h', 'volt_rolling_min_24h', 'volt_rolling_min_6h', 'volt_rolling_std_12h',
    'volt_rolling_std_24h', 'volt_rolling_std_6h', 'model'
]

# --- Helper Function for RUL Calculation ---
def calculate_rul(failure_probability):
    """
    Calculates Remaining Useful Life (RUL) percentage based on predicted failure probability.

    Args:
        failure_probability: Predicted probability of failure (float between 0 and 1).

    Returns:
        Remaining Useful Life (RUL) percentage (float between 0 and 1).
    """
    rul = 1 - failure_probability
    return max(0, min(1, rul))  # Ensure RUL is between 0 and 1

# --- App Title and Description ---
st.title("⚙️ Predictive Maintenance: Failure Predictor")

st.markdown("""
This application uses a pre-trained **Random Forest Classifier** to predict the likelihood of a machine component failure based on simulated sensor data. The model was trained on the "Predictive Maintenance" dataset, which includes telemetry data (voltage, rotation, pressure, vibration), error logs, maintenance records, and machine metadata.

**How it Works:**
1. Click the **"Generate & Predict"** button below.
2.  The app generates a random but realistic set of sensor readings and machine features.
3.  These features are fed into the trained model.
4.  The model outputs a **failure probability**, indicating how likely the machine is to fail in the near future.

---
""")

# --- App Sidebar with Details ---
with st.sidebar:
    st.header("Project Details")
    st.markdown("""
    ### Model Information
    - **Model Type:** Random Forest Classifier
    - **Prediction Target:** Imminent Failure (Yes/No)
    - **Features Used:** 79 engineered features from sensor, time, and machine data.

    ### Data Source
    This app uses a model trained on the public [Microsoft Azure Predictive Maintenance](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance) dataset.

    #### Original Data:
    -  **Telemetry:** Hourly `volt`, `rotate`, `pressure`, `vibration`.
    - **Errors:** Error codes recorded for machines.
    - **Maintenance:** Component replacement records.
    - **Machines:** `age` and `model` type.

    #### Feature Engineering
    The model's accuracy is enhanced by creating features that capture trends over time:
    - **Rolling Features:** Mean, standard deviation, min, and max over 6, 12, and 24-hour windows.
    - **Lag Features:** Sensor values from 1, 3, 6, and 12 hours prior.
    - **Time Features:** Cyclical representations of hour and day.
    """)

st.header("Live Failure Prediction")

col1, col2 = st.columns([1, 3])

with col1:
    if st.button("Generate & Predict", type="primary", use_container_width=True):
        # NOTE: The data ranges are ESTIMATES. For better results, these should be
        # updated with the .describe() output from your final training data.
        data_ranges = {
            'age': (0, 20), 'day_cos': (-1, 1), 'day_of_month': (1, 31), 'day_of_week': (0, 6), 'day_sin': (-1, 1),
            'hour': (0, 23), 'hour_cos': (-1, 1), 'hour_sin': (-1, 1), 'month': (1, 12), 'model': (0, 3), # Model is label encoded
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

        # Generate random data for each feature based on its defined range
        rand_values = [
            np.random.uniform(low, high) if feat != 'model' else np.random.randint(low, high + 1)
            for feat, (low, high) in data_ranges.items()
        ]
        
        # Create a DataFrame from the generated values
        rand_data = pd.DataFrame([rand_values], columns=feature_columns)
        
        # Make prediction using the loaded model
        # Use predict_proba to get the probability of failure (class 1)
        pred_proba = model.predict_proba(rand_data)[0]
        failure_probability = pred_proba[1]

        # Calculate RUL
        rul_percentage = calculate_rul(failure_probability)
        
        # Store results in session state to display them
        st.session_state['rul_percentage'] = rul_percentage
        st.session_state['rand_data'] = rand_data

        # --- Lifespan Input and RUL Calculation ---
        st.write("---")
        st.write("#### Calculate RUL from Expected Lifespan")
        col3, col4 = st.columns(2)
        with col3:
            lifespan = st.number_input("Enter expected lifespan:", min_value=1, step=1)
        with col4:
            unit = st.selectbox("Select unit:", ["Hours", "Months", "Years"])

        if st.button("Calculate RUL from Lifespan", use_container_width=True):
            if unit == "Hours":
                estimated_rul = lifespan * rul_percentage
            elif unit == "Months":
                estimated_rul = (lifespan * rul_percentage)
            else:  # Years
                estimated_rul = (lifespan * rul_percentage)
            st.success(f"Estimated RUL: {estimated_rul:.2f} {unit.lower()}")

with col2:
    if 'rul_percentage' in st.session_state:
        rul_percentage = st.session_state['rul_percentage']

        # Display the prediction with a clear status message
        st.write("#### Prediction Result: Remaining Useful Life (RUL)")
        if rul_percentage < 0.3:
            st.error(f"**Status:** Critical (RUL: {rul_percentage:.0%})", icon="🚨")
            interpretation = "The model predicts a **critical RUL**, indicating imminent failure. Immediate maintenance is required."
        elif rul_percentage < 0.7:
            st.warning(f"**Status:** Approaching Maintenance (RUL: {rul_percentage:.0%})", icon="⚠️")
            interpretation = "The model suggests the machine is **approaching the need for maintenance**. Please schedule maintenance soon."
        else:
            st.success(f"**Status:** Operational (RUL: {rul_percentage:.0%})", icon="✅")
            interpretation = "The model predicts the machine is in **good operational condition**. No immediate maintenance is needed."
        st.info(interpretation)

        # Show the generated data in an expander
        with st.expander("View Generated Sensor Data"):
            st.dataframe(st.session_state['rand_data'])
    else:
        st.info("Click the 'Generate & Predict' button to see the model's prediction.")
