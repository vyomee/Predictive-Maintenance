import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Predictive Maintenance: RUL Estimator",
    page_icon="🛠️",
    layout="wide",
)

# --- Load Model ---
# Ensure you have your trained RandomForestClassifier model saved as 'random_forest_model.joblib'
try:
    model = joblib.load("random_forest_model.joblib")
    # feature_columns must be in the same order as the model was trained on
    feature_columns = model.feature_names_in_
except FileNotFoundError:
    st.error("Error: The model file 'random_forest_model.joblib' was not found.")
    st.info("Please make sure your trained Random Forest model is saved in the correct path and contains the feature names.")
    model = None
    feature_columns = [] # Set to empty if model fails to load


# --- UI Elements ---
st.title("⚙️ Remaining Useful Life (RUL) Estimator")
st.write(
    "This app uses a Random Forest model to predict the RUL of a machine. "
    "Enter the machine's expected lifespan and click the button to generate random sensor data and see the prediction."
)

st.sidebar.header("Machine Lifespan")
lifespan_value = st.sidebar.number_input(
    "Enter the expected total lifespan:", min_value=1.0, value=20000.0, step=1000.0, format="%.1f"
)
lifespan_unit = st.sidebar.selectbox(
    "Select the time unit for the lifespan:", ["Hours", "Months", "Years"]
)

st.divider()
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

# --- Live Prediction Section ---
st.header("Live Failure Prediction")

col1, col2 = st.columns([1, 3])

with col1:
    generate_button = st.button("Generate & Predict", type="primary", use_container_width=True)

with col2:
    if model and generate_button:
        # These data ranges are based on your provided snippet.
        # They should align with the .describe() output from your final training data.
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

        # Generate a dictionary of random values, ensuring feature order matches the model's training columns
        rand_values = {}
        for feat in feature_columns:
            if feat not in data_ranges:
                st.warning(f"Warning: Feature '{feat}' was in the model but not in the data_ranges dict. Using default value 0.")
                rand_values[feat] = 0
                continue

            low, high = data_ranges[feat]
            if feat in ['model', 'day_of_month', 'day_of_week', 'hour', 'month', 'quarter', 'age']:
                 rand_values[feat] = np.random.randint(low, high + 1)
            else:
                 rand_values[feat] = np.random.uniform(low, high)

        # Create a DataFrame from the generated values
        rand_data = pd.DataFrame([rand_values], columns=feature_columns)

        # Make prediction to get probabilities
        # predict_proba returns [[P(class_0), P(class_1)]]
        prediction_proba = model.predict_proba(rand_data)

        # RUL is the probability of "non-failure" (Class 0)
        rul_percentage = prediction_proba[0][0] * 100
        failure_probability = prediction_proba[0][1] * 100

        # Calculate RUL value based on user input
        rul_value = (rul_percentage / 100) * lifespan_value

        # --- Display Results ---
        st.subheader("Prediction Results")
        st.info(f"The model predicts a **{failure_probability:.2f}% probability of failure** based on this data.")

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(
                label="Remaining Useful Life (RUL)",
                value=f"{rul_percentage:.2f}%",
            )
        with res_col2:
            st.metric(
                label=f"RUL in {lifespan_unit}",
                value=f"{rul_value:.2f}",
            )

        with st.expander("View Generated Sensor Data"):
             st.dataframe(rand_data)
    elif not model:
         st.warning("Please load a model file to enable predictions.")
