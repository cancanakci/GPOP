import joblib
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

def check_input_values(input_df, training_data):
    """Check if input values are within the range of training data.
    
    Args:
        input_df: DataFrame with features to predict on (unscaled)
        training_data: Dictionary containing training data statistics (unscaled)
        
    Returns:
        tuple: (warning_flags_df, yellow_warnings, red_warnings, warning_ranges) where:
            - warning_flags_df is a DataFrame with boolean flags for each row and feature
            - yellow_warnings is a list of feature names with any yellow warnings
            - red_warnings is a list of feature names with any red warnings
            - warning_ranges is a dict with min/max ranges for features with warnings
    """
    yellow_warnings = []
    red_warnings = []
    warning_ranges = {}
    
    # Get training data statistics from the unscaled X_train
    X_train = training_data['X_train']
    
    # Create DataFrames to store warning flags
    red_flags = pd.DataFrame(False, index=input_df.index, columns=input_df.columns)
    yellow_flags = pd.DataFrame(False, index=input_df.index, columns=input_df.columns)
    
    for feature in input_df.columns:
        if feature not in X_train.columns:
            continue # Skip if the feature is not in training data
            
        # Get training data statistics
        train_min = X_train[feature].min()
        train_max = X_train[feature].max()
        
        # Store ranges for features with warnings
        warning_ranges[feature] = {
            'min': train_min,
            'max': train_max
        }
        
        # Check input values for each row
        for index, row in input_df.iterrows():
            input_value = row[feature]
            
            # Check for red warnings (outside min/max)
            if input_value < train_min or input_value > train_max:
                red_flags.loc[index, feature] = True
                # Only add the feature to red_warnings once
                if feature not in red_warnings:
                    red_warnings.append(feature)
    
    # Create a combined warning flags DataFrame
    warning_flags_df = pd.DataFrame({
        'has_red_warning': red_flags.any(axis=1),
        'has_yellow_warning': yellow_flags.any(axis=1),
        'red_warning_features': red_flags.apply(lambda x: x[x].index.tolist(), axis=1),
        'yellow_warning_features': yellow_flags.apply(lambda x: x[x].index.tolist(), axis=1)
    })
    
    return warning_flags_df, yellow_warnings, red_warnings, warning_ranges

def load_model(model_path):
    return joblib.load(model_path)

def predict(model, input_df):
    """Make predictions using the model.
    
    Args:
        model: The trained model
        input_df: DataFrame with features to predict on (scaled)
        
    Returns:
        numpy array of predictions
    """
    # Debug information
    print("Input DataFrame shape (scaled):", input_df.shape)
    print("Input DataFrame columns (scaled):", input_df.columns.tolist())
    # print("Input DataFrame values (scaled):", input_df.values) # Avoid printing potentially large arrays
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Clamp predictions below 0 to 0
    prediction = np.maximum(prediction, 0)
    
    # Debug information
    print("Raw prediction:", prediction)
    print("Prediction shape:", prediction.shape)
    
    return prediction
