# Predictive Maintenance - Remaining Useful Life (RUL) Estimator

This project demonstrates a predictive maintenance framework for estimating the Remaining Useful Life (RUL) of machines, using a pre-trained Random Forest model deployed with Streamlit.

## Overview

Predictive maintenance aims to estimate the remaining time a machine or component can operate before failure, enabling proactive maintenance strategies. This project focuses on utilizing sensor data to predict the RUL of machines.

The project leverages the following components:

*   **Dataset:** Simulated telemetry data (voltage, rotation, pressure, vibration) collected from machines.  Additional details and sources for this data are described in the Streamlit App.
*   **Model Training:**  Three models were trained and compared for the prediction of RUL based on preprocessed sensor data. The random forest turned out to be the best model. The original notebook details this process.
*   **Streamlit App:** A web application that allows users to input random sensor data and receive an RUL prediction from the trained model.

## Repository Contents
*   `streamlit_app.py`: The main Python script for the Streamlit application.
*   `random_forest_rul_model.joblib` The serialized Random Forest model file.
*   `README.md`: This file, providing an overview of the project.



## Getting Started

### Prerequisites

*   Python 3.6 or higher
*   Required Python packages:
    *   streamlit
    *   pandas
    *   scikit-learn 
    *   numpy

### Installation and Usage

1.  Clone this repository:

    ```bash
    git clone https://github.com/vyomee/Predictive-Maintenance/
    cd Predictive-Maintenance
    ```

2.  Install the required Python packages:

    ```bash
    pip install streamlit pandas scikit-learn numpy
    ```

3.  Run the Streamlit application:

    ```bash
    streamlit run streamlit_app.py
    ```

4.  Open your web browser and navigate to the address displayed in the terminal.

5.  In the Streamlit application, click the "Generate Random Data" button to generate random sensor data and see the predicted RUL.

## Data

The app comes with some description about the data, but to get access to the data it self, check the following site:

[Kaggle dataset: Microsoft Azure Predictive Maintenance](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)

## Model
A Random Forest model was chosen as the best model for this RUL prediction task based on the evaluation metrics F1 score, Accuracy, Area Under Curve (AUC), Precision and Recall
The model was trained on features derived from sensor data, including:
- Aggregated sensor readings over time windows (e.g., mean, standard deviation of voltage, rotation, pressure, vibration over the last 24 hours)
- Differences between current and historical sensor readings
- Machine metadata (age, model type)

## Additional Notes and Considerations

* The Streamlit app generates random sensor data within predefined ranges.
* The RUL prediction is based on a trained Random Forest model. The accuracy of the prediction depends on the quality and representativeness of the data used for training.


## Contributing

Feel free to contribute to this project by submitting issues or pull requests.



## License

This project is licensed under the [MIT License](LICENSE).
