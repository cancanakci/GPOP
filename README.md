# Geothermal Power Output Prediction (GPOP)

**GPOP** is a proof-of-concept machine learning application designed specifically for binary cycle geothermal power plant operators, engineers, and analysts. It leverages advanced time series forecasting and XGBoost regression to predict geothermal power output based on operational parameters, helping optimize plant performance and inform strategic decisions about well drilling and maintenance.


### 🎯 **Core Purpose**
GPOP helps geothermal power plant operators:
- **Predict future power output** based on current operational conditions
- **Optimize production planning** by understanding how different parameters affect power generation
- **Plan strategic interventions** like make-up well drilling to maintain production levels
- **Analyze long-term production trends** and their impact on plant profitability
- **Identify operational anomalies** through data analysis and visualization

### 🏭 **Real-World Applications**

1. **Daily Operations**: Predict tomorrow's power output based on current plant conditions
2. **Strategic Planning**: Model long-term production decline and plan interventions
3. **Well Drilling Decisions**: Simulate the impact of new make-up wells on production
4. **Performance Monitoring**: Track key performance indicators and identify optimization opportunities
5. **Risk Assessment**: Understand how changes in operational parameters affect output reliability

### 🔧 **Key Operational Parameters**
The app typically works with standard geothermal plant data including:
- Brine flowrate (T/h)
- Steam/NCG flowrate (T/h) 
- Ambient temperature (°C)
- Heat exchanger pressure differentials (Bar)
- Wellhead temperatures and pressures
- Other plant-specific operational metrics

## Features

### 📊 **Model Training & Prediction**
-   **Pre-trained Model**: Comes with a default model trained on 4 years of representative geothermal data for immediate use
-   **Custom Model Training**: Upload your own plant data to create tailored prediction models
-   **Single & Batch Predictions**: Get instant predictions for specific conditions or process entire datasets
-   **Data Quality Checks**: Built-in validation to flag unusual input values that may affect prediction accuracy

### 📈 **Time Series Analysis**
-   **Historical Data Visualization**: Comprehensive charts showing trends, correlations, and seasonal patterns
-   **Feature Importance Analysis**: Understand which operational parameters most significantly impact power output
-   **Data Cleaning Tools**: Automatic detection and handling of outliers, missing values, and inconsistencies
-   **Statistical Insights**: Detailed analysis of data distributions and time series characteristics

### 🔮 **Long Horizon Extrapolation**
-   **Multi-Year Forecasting**: Project power output trends up to 40 years into the future
-   **Flexible Trend Modeling**: Apply different mathematical models (constant, linear, exponential, polynomial) to each operational parameter
-   **Seasonality Integration**: Automatically incorporate yearly seasonal patterns into long-term projections
-   **Scenario Planning**: Compare different operational strategies and their long-term impacts

### ⛏️ **Make-Up Well (MUW) Drilling Simulation**
-   **Production Decline Modeling**: Simulate natural decline in geothermal production over time
-   **Automated Drilling Triggers**: Set power output thresholds that automatically trigger new well drilling in simulations
-   **Well Performance Modeling**: Define flowrate, steam/brine composition, and performance characteristics of new wells
-   **Economic Impact Analysis**: Understand how strategic well drilling affects long-term production curves
-   **ROI Planning**: Compare scenarios with and without intervention to inform investment decisions

### 🌅 **Next Day Prediction**
-   **Short-term Forecasting**: Specialized module for predicting next-day power output
-   **Baseline Comparisons**: Compare ML predictions against simple statistical baselines
-   **Operational Planning**: Help operators prepare for expected production levels and identify potential issues

## Technical Architecture

### 🧠 **Machine Learning Stack**
- **XGBoost Regression**: Primary prediction algorithm optimized for time series data
- **Standard Scaling**: Ensures all features contribute appropriately to predictions
- **Cross-Validation**: Robust model evaluation using 5-fold cross-validation
- **Hyperparameter Optimization**: Grid search capabilities for model tuning

### 🎨 **User Interface**
- **Streamlit Framework**: Interactive web application
- **Plotly Visualizations**: Interactive charts and graphs for data exploration
- **Tabular Data Display**: Presentation of results and datasets
- **File Upload/Download**: Easy data import/export functionality

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

## Setup & Installation

1.  Clone the repository or download the project folder.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  The app comes with a pre-trained default model. To run the app, use the following command:
    ```bash
    streamlit run src/app.py
    ```

## Usage Guide

### Getting Started
1. **Launch the Application**: Run `streamlit run src/app.py`
2. **Choose Your Approach**:
   - Use the pre-trained default model for immediate predictions
   - Upload your own data to then retrain a custom model either through the app or through `streamlit run src/create_defailt_model.py`
3. **Explore Your Data**: Use the analysis tabs to understand your data patterns
4. **Make Predictions**: Input current conditions to get power output forecasts
5. **Plan for the Future**: Use long-term forecasting to inform strategic decisions regarding MUWs.

### Data Requirements
Your data should be in CSV or Excel format with:
- **Timestamp Column**: Date/time information for each measurement
- **Operational Parameters**: Flowrates, temperatures, pressures, etc.
- **Target Variable**: Historical power output (for training)
- **Regular Intervals**: Consistent time spacing between measurements (hourly recommended)

### Best Practices
- **Data Quality**: Ensure your data is clean with minimal gaps
- **Feature Selection**: Include all relevant operational parameters
- **Validation**: Always review prediction warnings and validate results against operational knowledge
- **Regular Updates**: Retrain models periodically with new data to maintain accuracy


For technical support or customization requests, refer to the documentation in the `report/` directory or contact me at icancanakci@hotmail.com