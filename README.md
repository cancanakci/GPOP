# Geothermal Power Output Prediction (GPOP)

GPOP is a machine learning-based tool for predicting power output from geothermal power plants. It provides a comprehensive interface for model training, batch/single predictions, and advanced time series forecasting with a make-up well (MUW) drilling simulation.

## Features

-   **Pre-trained Model**: Comes with a default model for immediate predictions.
-   **Custom Model Training**: Train new models on your own data with a user-friendly interface.
-   **Data Processing**: Includes tools for data parsing, cleaning, and outlier detection.
-   **Exploratory Data Analysis (EDA)**: Visualizations for feature importance, correlation, and distributions.
-   **Singular and Batch Predictions**: Make single predictions or upload a file for batch processing.
-   **Time Series Forecasting**: Project feature trends into the future and predict long-term power output.
    -   **Flexible Trend Modeling**: Apply constant, linear, exponential, or polynomial trends to each feature.
    -   **Seasonality**: Automatically detect and apply yearly seasonality to projections.
-   **Make-Up Well (MUW) Simulation**:
    -   Simulate the impact of drilling new make-up wells to counteract production decline.
    -   Set a power output threshold to trigger drilling automatically.
    -   Define the flowrate and steam/brine composition of new wells.
    -   The simulation models the production uplift and its subsequent decay based on the defined feature trends.

## Setup

1.  Clone the repository or download the project folder.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  The app comes with a pre-trained default model. To run the app, use the following command:
    ```bash
    streamlit run src/app.py
    ```

## Project Structure

-   `data/` - Holds the default data and other data files.
-   `models/` - Stores saved model files (`.joblib`), including the default model.
-   `src/` - Contains all Python source code.
    -   `app.py`: The main Streamlit application file.
    -   `core.py`: Core logic for forecasting and MUW simulation.
    -   `train.py`: Model training pipeline.
    -   `predict.py`: Prediction functions.
    -   `data_processing.py`: Data loading and preprocessing functions.
    -   `ui_components.py`: Functions for creating UI elements and plots.
    -   `file_utils.py`: Utilities for file and model handling.

## Usage

The application is organized into two main tabs: "Model Training & Prediction" and "Time Series Forecasting".

### Model Training & Prediction

-   **Use Default Model**: Load the pre-trained model to start making predictions right away.
-   **Train New Model**:
    1.  Upload your training data (CSV or Excel).
    2.  Configure time series settings (e.g., datetime column, data frequency).
    3.  Map your data columns to the required inputs (e.g., target, brine flow, steam flow).
    4.  Adjust model hyperparameters and train the model.
    5.  The newly trained model becomes available for predictions and forecasting.

### Time Series Forecasting

This tab becomes active once a model is loaded (either default or newly trained).

1.  **Configure Projections**:
    -   Set the number of years to project into the future.
    -   For each feature, define a trend model (Constant, Linear, Exponential, Polynomial) and whether to include seasonality.
2.  **Generate Scenario**: Creates a baseline forecast based on your trend configurations.
3.  **Run MUW Simulation**:
    -   Define a "drilling threshold" (in MW). If the yearly average power output drops below this value, a new well is simulated.
    -   Specify the flowrate and steam percentage for the new make-up wells.
    -   The simulation adjusts the flowrate features and re-predicts the power output, showing the impact of the new wells.
    -   Results are visualized, comparing the original prediction with the adjusted one. You can also download the simulation data.

## Data Format

Your data should be in CSV or Excel format. For best results, it should contain a datetime column and the features used by the model (e.g., Brine Flowrate, Steam Flowrate, Ambient Temperature, etc.).