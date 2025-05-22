import joblib
import pandas as pd
import numpy as np

def load_model(model_path):
    return joblib.load(model_path)

def predict(model, input_df):
    """Make predictions using the model.
    
    Args:
        model: The trained model
        input_df: DataFrame with features to predict on
        
    Returns:
        numpy array of predictions
    """
    # Debug information
    print("Input DataFrame shape:", input_df.shape)
    print("Input DataFrame columns:", input_df.columns.tolist())
    print("Input DataFrame values:", input_df.values)
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Debug information
    print("Raw prediction:", prediction)
    print("Prediction shape:", prediction.shape)
    
    return prediction
