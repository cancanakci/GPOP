import os
from train import train_model
import joblib
import pandas as pd
import logging
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_default_model():
    """Create and save the default model using the provided training data, with short hyperparameter tuning."""
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Path to the default training data
    default_data_path = "data/default_data.xlsx"
    
    if not os.path.exists(default_data_path):
        raise FileNotFoundError(f"Default training data not found at {default_data_path}")
    
    try:
        # Determine the datetime column from the default data file
        datetime_col = pd.read_excel(default_data_path, nrows=0).columns[0]
        
        # Load the data (no need to drop pressure column)
        df = pd.read_excel(default_data_path)
        # Save the data to a temporary file
        temp_path = "data/default_data_temp.xlsx"
        df.to_excel(temp_path, index=False)
        
        # Load and preprocess data for tuning
        from data_processing import load_and_parse
        data = load_and_parse(temp_path, datetime_col=datetime_col)
        target_column = data.columns[-1]
        X = data.drop(target_column, axis=1).select_dtypes(include='number')
        y = data[target_column]
        
        # Short hyperparameter tuning
        logger.info("Starting short hyperparameter tuning for default model...")
        param_grid = {
            'n_estimators': [500, 600, 700],
            'max_depth': [3, 5],
            'min_child_weight': [3, 5],
            'learning_rate': [0.03],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'random_state': [42],
        }
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, eval_metric='rmse')
        grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='neg_root_mean_squared_error', verbose=1)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        logger.info(f"Best parameters found: {best_params}")
        
        # Train the model, providing the datetime column and best params
        metrics = train_model(
            temp_path, 
            "models", 
            datetime_col=datetime_col,
            is_default=True,
            test_size=0.4,
            model_params=best_params
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