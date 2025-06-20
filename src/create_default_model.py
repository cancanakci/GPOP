"""
create_default_model.py
-----------------------
Script to create the default model using the provided dataset.
"""

import os
import sys
import pandas as pd
from train import train_model, train_nextday_model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Create default model and next-day prediction model."""
    
    # Check if data file exists
    data_file = "data/default_data.xlsx"
    if not os.path.exists(data_file):
        logger.error(f"Data file {data_file} not found!")
        sys.exit(1)
    
    # Create models directory if it doesn't exist
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"Created models directory: {models_dir}")
    
    try:
        # Load data to get column names
        df = pd.read_excel(data_file)
        datetime_col = df.columns[0]  # First column is datetime
        target_col = "Gross Power (MW)"  # Assuming this is the target
        
        logger.info("Starting default model training...")
        
        # Train default model
        default_metrics = train_model(
            data_source=data_file,
            models_dir=models_dir,
            target_column=target_col,
            datetime_col=datetime_col,
            is_default=True
        )
        
        logger.info("Default model training completed!")
        logger.info(f"Default model R²: {default_metrics['metrics']['r2']:.4f}")
        logger.info(f"Default model RMSE: {default_metrics['metrics']['rmse']:.4f}")
        
        # Train next-day prediction model
        logger.info("Starting next-day prediction model training...")
        
        nextday_metrics = train_nextday_model(
            data_source=data_file,
            models_dir=models_dir,
            target_column=target_col,
            datetime_col=datetime_col,
            window_hours=24
        )
        
        logger.info("Next-day prediction model training completed!")
        logger.info(f"Next-day model R²: {nextday_metrics['overall_metrics']['r2']:.4f}")
        logger.info(f"Next-day model RMSE: {nextday_metrics['overall_metrics']['rmse']:.4f}")
        
        logger.info("All models created successfully!")
        
    except Exception as e:
        logger.error(f"Error creating models: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()