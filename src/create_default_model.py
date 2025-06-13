import os
from train import train_model
import joblib

def create_default_model():
    """Create and save the default model using the provided training data."""
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Path to the default training data
    default_data_path = "data/default_data.xlsx"
    
    if not os.path.exists(default_data_path):
        raise FileNotFoundError(f"Default training data not found at {default_data_path}")
    
    try:
        # Train the model
        metrics = train_model(default_data_path, "models", is_default=True)
        print("\nDefault model created successfully!")
        print("\nModel Performance Summary:")
        print(f"R² Score: {metrics['metrics']['r2']:.4f}")
        print(f"RMSE: {metrics['metrics']['rmse']:.4f}")
        print(f"Cross-validation R²: {metrics['metrics']['cv_metrics']['r2_mean']:.4f} (±{metrics['metrics']['cv_metrics']['r2_std'] * 2:.4f})")
    except Exception as e:
        print(f"Error creating default model: {e}")

if __name__ == "__main__":
    create_default_model() 