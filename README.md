# Geothermal Power Output Prediction Tool

This project is a machine learning-based tool for predicting power output from geothermal systems. It includes data processing, model training, and a Streamlit web app for interactive predictions.

## Features
- Pre-trained default model for immediate predictions
- Option to train custom models with your own data
- Data preprocessing and feature engineering
- Model training and evaluation
- Power output prediction
- Streamlit web app for interactive predictions

## Setup
1. Clone the repository or download the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create the default model:
   ```bash
   python src/create_default_model.py
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run src/app.py
   ```

## Project Structure
- `data/` - Raw and processed data
- `notebooks/` - Jupyter notebooks for exploration
- `src/` - Source code (data processing, training, prediction, app)
- `models/` - Saved models (including default model)

## Usage
The app provides two main options:
1. Use Default Model:
   - Use the pre-trained model for immediate predictions
   - Upload your data file to make predictions

2. Train New Model:
   - Upload your own training data
   - Train a new model with your data
   - The app will automatically switch to using your newly trained model
   - Make predictions with your custom model

## Data Format
Your data should be in CSV or Excel format and include the target feature "Brüt Güç" (Gross Power) along with other relevant features.