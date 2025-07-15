import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading the pre-trained models

# Load the pre-trained Random Forest model
model = joblib.load("random_forest_rul_model.joblib")  # Replace with the actual path to your model file

# Example feature columns (replace with your actual feature names)
feature_columns = ['voltmean_24h', 'rotatemean_24h', 'pressuremean_24h', 'vibrationmean_24h', 'voltsd_24h', 'rotatesd_24h', 'pressuresd_24h', 'vibrationsd_24h', 'voltmean_diff', 'rotatemean_diff', 'pressuremean_diff', 'vibrationmean_diff', 'age', 'model_type']

st.title("Predictive Maintenance - RUL Estimator")

st.markdown("""
### Dataset Info
This an example data source which can be used for Predictive Maintenance Model Building. It consists of the following data:

~Machine conditions and usage: The operating conditions of a machine e.g. data collected from sensors.
~Failure history: The failure history of a machine or component within the machine.
~Maintenance history: The repair history of a machine, e.g. error codes, previous maintenance activities or component replacements.
~Machine features: The features of a machine, e.g. engine size, make and model, location.

Data files:
~Telemetry Time Series Data (PdM_telemetry.csv): It consists of hourly average of voltage, rotation, pressure, vibration collected from 100 machines for the year 2015.

~Error (PdM_errors.csv): These are errors encountered by the machines while in operating condition. Since, these errors don't shut down the machines, these are not considered as failures. The error date and times are rounded to the closest hour since the telemetry data is collected at an hourly rate.

~Maintenance (PdM_maint.csv): If a component of a machine is replaced, that is captured as a record in this table. Components are replaced under two situations: 1. During the regular scheduled visit, the technician replaced it (Proactive Maintenance) 2. A component breaks down and then the technician does an unscheduled maintenance to replace the component (Reactive Maintenance). This is considered as a failure and corresponding data is captured under Failures. Maintenance data has both 2014 and 2015 records. This data is rounded to the closest hour since the telemetry data is collected at an hourly rate.

~Failures (PdM_failures.csv): Each record represents replacement of a component due to failure. This data is a subset of Maintenance data. This data is rounded to the closest hour since the telemetry data is collected at an hourly rate.

~Metadata of Machines (PdM_Machines.csv): Model type & age of the Machines.

### Exploratory Analysis
The model was trained on preprocessed and normalized data using classical ML and transformer-based models.
~ Temporal Pattern Analysis
~ Sensor data Statistical Analysis
~ Machine and Component Analysis
~ Maintenance Events Analysis
~ Failure Events Analysis
~ Correlation between sensors

### Data pre-processing and Feature Engineering
~ Data Integration and Merging
~ Creating Failure Labels Using Maintenance Records
~ Feature Engineering for Time Series Data
~ Data Quality Assessment and Cleaning
~ Time aware data splitting
~ Feature scaling and Normalization

### Model Training
~ Logistic regression
~ Random Forest Model
~ LSTM (Long Short Term Memory) Model

### Model Evaluation and Comparison
~ F1 score
~ Accuracy
~ Area Under Curve (AUC)
~ Precision and Recall 
""")

st.markdown("---")

st.header("Generate Random Sensor Data")

# Define the range for random data generation based on the dataset (example ranges, adjust accordingly)
data_ranges = {
    'voltmean_24h': (150, 180),  # Example: Voltage mean in the last 24 hours
    'rotatemean_24h': (400, 500), # Example: Rotation speed mean in the last 24 hours
    'pressuremean_24h': (90, 110), # Example: Pressure mean in the last 24 hours
    'vibrationmean_24h': (35, 50),  # Example: Vibration mean in the last 24 hours
    'voltsd_24h': (5, 15),        # Example: Voltage standard deviation in the last 24 hours
    'rotatesd_24h': (20, 50),      # Example: Rotation speed standard deviation
    'pressuresd_24h': (5, 15),     # Example: Pressure standard deviation
    'vibrationsd_24h': (2, 8),      # Example: Vibration standard deviation
    'voltmean_diff': (-5, 5),      # Example: Difference in voltage mean
    'rotatemean_diff': (-20, 20),   # Example: Difference in rotation speed mean
    'pressuremean_diff': (-2, 2),   # Example: Difference in pressure mean
    'vibrationmean_diff': (-1, 1),   # Example: Difference in vibration mean
    'age': (0, 20),                # Example: Machine age
    'model_type': (1, 4)           # Example: Machine model type (assuming categorical 1-4)
}

if st.button("Generate Random Data"):
    # Generate random data within the specified range
    rand_values = np.random.uniform(min_value, max_value, len(feature_columns))
    rand_data = pd.DataFrame([rand_values], columns=feature_columns)

    # Make prediction using the loaded model
    prediction = model.predict(rand_data)[0]

    st.write("### Generated Sensor Data:")
    st.write(rand_data)

    st.write("---")

    st.write("### Model Output")

    # Provide some interpretation of the output (adjust based on your model and data)
    if prediction > 100:
        interpretation = "The model predicts a high RUL, indicating the machine is likely to function for an extended period."
    elif prediction > 50:
        interpretation = "The model predicts a moderate RUL, suggesting the machine has some remaining operational life but may require attention soon."
    else:
        interpretation = "The model predicts a low RUL, indicating the machine may be nearing the end of its useful life and maintenance should be scheduled."

    st.success(f"Predicted RUL: {round(prediction, 2)} hours.")
    st.info(interpretation)