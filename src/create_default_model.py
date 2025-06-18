import os
from train import train_model
import joblib
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_default_model():
    """Create and save the default model using the provided training data."""
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Path to the default training data
    default_data_path = "data/default_data.xlsx"
    
    if not os.path.exists(default_data_path):
        raise FileNotFoundError(f"Default training data not found at {default_data_path}")
    
    try:
        # Determine the datetime column from the default data file
        datetime_col = pd.read_excel(default_data_path, nrows=0).columns[0]
        
        # Load the data and drop the pressure column
        df = pd.read_excel(default_data_path)
        if 'Heat Exchanger Pressure Differential (Bar)' in df.columns:
            df = df.drop(columns=['Heat Exchanger Pressure Differential (Bar)'])
        # Save the modified data to a temporary file
        temp_path = "data/default_data_nopressure_temp.xlsx"
        df.to_excel(temp_path, index=False)
        
        # Train the model, providing the datetime column
        metrics = train_model(
            temp_path, 
            "models", 
            datetime_col=datetime_col,
            is_default=True,
            test_size=0.4
        )
        logger.info("Default model created successfully!")
        logger.info("Model Performance Summary:")
        logger.info(f"R² Score: {metrics['metrics']['r2']:.4f}")
        logger.info(f"RMSE: {metrics['metrics']['rmse']:.4f}")
        logger.info(f"Cross-validation R²: {metrics['metrics']['cv_metrics']['r2_mean']:.4f} (±{metrics['metrics']['cv_metrics']['r2_std'] * 2:.4f})")
        # Optionally, remove the temp file after training
        os.remove(temp_path)
    except Exception as e:
        logger.error(f"Error creating default model: {e}")
        raise

if __name__ == "__main__":
    create_default_model() 