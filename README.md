# Geothermal Power Output Prediction Tool

GPOP is a machine learning-based tool for predicting power output or any other target from geothermal systems. It includes data processing, model training and prediction capabilities, time series prediction with custom modifiers through a Streamlit web app.

## Features
- Pre-trained default model for immediate predictions
- Option to train custom models with your own data
- Data preprocessing, EDA
- Model training and evaluation
- Singular and batch predictions
- Time series tool for extrapolation with flexible modifiers

## Setup
1. Clone the repository or download the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create the default model (optional, see below for custom target):
   ```bash
   python src/create_default_model.py
   ```
   - By default, the last column in your data will be used as the target. To specify a different target column, edit `create_default_model.py` and pass the column name to `create_default_model(target_column="Your Target")`.
4. Run the Streamlit app:
   ```bash
   streamlit run src/app.py
   ```

## Project Structure
- `data/` - Raw and processed data
- `notebooks/` - Jupyter notebooks exploring concepts, ideas and avoided pitfalls
- `src/` - Source code (data processing, training, prediction, time series, app)
- `models/` - Saved models (including default model)

## Usage
The app provides two main options:
1. Use Default Model:
   - Use the pre-trained model for immediate predictions
   - The default model uses Gross Power (MW) as target

2. Train New Model:
   - Upload your own training data
   - Train a new model with your data, dynamically select target for predictions & time series
   - Make predictions with custom model

## Data Format
Your data should be in CSV or Excel format and include the target feature you wish to predict (e.g., "Gross Power", "Reinjection Temperature (°C)", etc.) along with other relevant features. You can train your own model for any target column present in your data.