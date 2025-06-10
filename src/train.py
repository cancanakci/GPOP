import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
import joblib
from data_prep import load_data, preprocess_data
import os
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
import json

def train_model(data_path, models_dir, target_column=None, n_splits=5, is_default=False, model_params=None):
    """Loads data, preprocesses it, trains XGBoost model with cross-validation, and saves model with metrics."""
    
    # Create timestamp for model versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create model paths with timestamps or default names
    if is_default:
        model_path = os.path.join(models_dir, "default_model.pkl")
        scaler_path = os.path.join(models_dir, "default_scaler.pkl")
        feature_names_path = os.path.join(models_dir, "default_feature_names.pkl")
        metrics_path = os.path.join(models_dir, "default_metrics.json")
        training_data_path = os.path.join(models_dir, "default_training_data.pkl")
    else:
        model_path = os.path.join(models_dir, f"xgboost_{timestamp}.pkl")
        scaler_path = os.path.join(models_dir, f"scaler_{timestamp}.pkl")
        feature_names_path = os.path.join(models_dir, f"feature_names_{timestamp}.pkl")
        metrics_path = os.path.join(models_dir, f"metrics_{timestamp}.json")
        training_data_path = os.path.join(models_dir, f"training_data_{timestamp}.pkl")
    
    # Load data - data_path can be either a file path or a file object
    df = load_data(data_path)

    # --- Preprocessing (Interpolation) ---
    df_processed = preprocess_data(df)
    
    # If target_column is not specified, use the last column
    if target_column is None:
        target_column = df_processed.columns[-1]
    
    if target_column not in df_processed.columns:
        raise ValueError(f"Target column '{target_column}' not found in data after preprocessing.")

    # Separate features and target
    X = df_processed.drop(target_column, axis=1)
    y = df_processed[target_column]

    # Feature Engineering
    # 1. Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # 2. Select numeric features
    X = X.select_dtypes(include=['number'])
    
    if X.empty:
         raise ValueError("No numeric features available after preprocessing.")

    # Save the list of feature names in the correct order BEFORE scaling
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, feature_names_path)
    print(f"Feature names saved to {feature_names_path}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Scaling (Fit only on training data) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames with feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    print("Scaled features using StandardScaler.")

    # --- Model Training and Evaluation ---
    print("\nTraining XGBoostRegressor...")
    
    # Use custom parameters if provided, otherwise use defaults
    default_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'rmse'
    }
    
    # Update default parameters with custom parameters if provided
    if model_params:
        default_params.update(model_params)
    
    xgb_model = xgb.XGBRegressor(**default_params)

    # Perform k-fold cross-validation
    print(f"\nPerforming {n_splits}-fold cross-validation...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Cross-validation scores for different metrics
    cv_r2_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=kf, scoring='r2')
    cv_mse_scores = -cross_val_score(xgb_model, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')
    cv_rmse_scores = np.sqrt(cv_mse_scores)
    
    print(f"Cross-validation R² scores: {cv_r2_scores}")
    print(f"Mean CV R²: {cv_r2_scores.mean():.4f} (+/- {cv_r2_scores.std() * 2:.4f})")
    print(f"Mean CV RMSE: {cv_rmse_scores.mean():.4f} (+/- {cv_rmse_scores.std() * 2:.4f})")

    # Train final model on full training set
    xgb_model.fit(X_train_scaled, y_train, 
                 eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
                 verbose=False)
    
    # Get training history
    results = xgb_model.evals_result()
    train_loss = results['validation_0']['rmse']
    test_loss = results['validation_1']['rmse']
    
    xgb_preds = xgb_model.predict(X_test_scaled)
    xgb_mse = mean_squared_error(y_test, xgb_preds)
    xgb_rmse = np.sqrt(xgb_mse)
    xgb_r2 = r2_score(y_test, xgb_preds)
    print(f"\nFinal Test Set Performance:")
    print(f"XGBoost Test MSE: {xgb_mse:.4f}")
    print(f"XGBoost Test RMSE: {xgb_rmse:.4f}")
    print(f"XGBoost Test R-squared: {xgb_r2:.4f}")

    # Plot Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, xgb_preds, alpha=0.5, color='orange')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel("Actual Brüt Güç")
    plt.ylabel("Predicted Brüt Güç")
    plt.title("XGBoost Actual vs Predicted")
    plot_path = os.path.join(models_dir, f"actual_vs_predicted_{timestamp}.png")
    plt.savefig(plot_path)
    print(f"Actual vs Predicted plot saved to {plot_path}")
    plt.close()

    # --- Save model and scaler ---
    print("\nSaving model and scaler...")
    joblib.dump(xgb_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Save the training data
    training_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names
    }
    joblib.dump(training_data, training_data_path)
    
    print(f"XGBoost model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Training data saved to {training_data_path}")
    
    # Save metrics
    metrics = {
        'timestamp': timestamp,
        'model_type': 'XGBoost',
        'metrics': {
            'mse': float(xgb_mse),
            'rmse': float(xgb_rmse),
            'r2': float(xgb_r2),
            'cv_metrics': {
                'r2_mean': float(cv_r2_scores.mean()),
                'r2_std': float(cv_r2_scores.std()),
                'rmse_mean': float(cv_rmse_scores.mean()),
                'rmse_std': float(cv_rmse_scores.std()),
                'cv_scores': {
                    'r2': cv_r2_scores.tolist(),
                    'rmse': cv_rmse_scores.tolist()
                }
            },
            'loss_curve': {
                'train_loss': train_loss,
                'test_loss': test_loss
            }
        },
        'model_params': xgb_model.get_params(),
        'feature_names': list(feature_names),
        'plot_path': plot_path,
        'actual': y_test.tolist(),
        'predicted': xgb_preds.tolist(),
        'training_data_path': training_data_path
    }
    
    # Save metrics to JSON file
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Model metrics saved to {metrics_path}")
    
    return metrics

if __name__ == "__main__":
    data_file = "data/Can veriler.xlsx"
    models_dir = "models"
    
    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)

    try:
        metrics = train_model(data_file, models_dir)
        print("\nTraining completed successfully.")
        print("\nModel Performance Summary:")
        print(f"XGBoost R²: {metrics['metrics']['r2']:.4f}")
        print(f"Cross-validation R²: {metrics['metrics']['cv_metrics']['r2_mean']:.4f} (+/- {metrics['metrics']['cv_metrics']['r2_std'] * 2:.4f})")
    except Exception as e:
        print(f"Training failed: {e}")
