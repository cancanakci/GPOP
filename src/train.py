import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
import joblib
from data_processing import load_and_parse, create_nextday_features
import os
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(data_source, models_dir, target_column=None, datetime_col=None, start_date=None, freq=None, n_splits=5, is_default=False, model_params=None, test_size=0.2):
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
    
    # Validate that we have a method to create a datetime index
    if datetime_col is None and start_date is None:
        raise ValueError("Either 'datetime_col' or 'start_date' must be provided to train_model.")

    df = load_and_parse(data_source, datetime_col=datetime_col, start_date=start_date, freq=freq, silent=False)

    # --- Preprocessing (Interpolation) ---
    # Replicate the simple interpolation from the old preprocess_data function
    df_processed = df.interpolate(method='time').bfill()
    
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
    logger.info(f"Feature names saved to {feature_names_path}")

    # DEFAULT SPLIT IS 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # --- Scaling (Fit only on training data) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames with feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    logger.info("Scaled features using StandardScaler.")

    # --- Model Training and Evaluation ---
    logger.info("Training XGBoostRegressor...")
    
    # Use custom parameters if provided, otherwise use defaults
    default_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'rmse',
        'early_stopping_rounds': 50
    }
    
    if is_default:
        # The grid search functionality has been removed for the default model
        # to ensure a simpler and faster process. The model will now train
        # using a predefined set of optimized default parameters.
        logger.info("Skipping hyperparameter tuning for the default model.")

    # Update default parameters with custom parameters if provided
    if model_params:
        default_params.update(model_params)
    
    xgb_model = xgb.XGBRegressor(**default_params)

    # --- Cross-validation ---
    logger.info("Performing 5-fold cross-validation...")
    
    # Create a new dictionary of parameters without early stopping for CV
    cv_params = xgb_model.get_params()
    cv_params.pop('early_stopping_rounds', None)
    cv_model = xgb.XGBRegressor(**cv_params)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=12345)
    cv_r2_scores = cross_val_score(cv_model, X_train_scaled, y_train, cv=kf, scoring='r2')
    cv_mse_scores = -cross_val_score(cv_model, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')
    cv_rmse_scores = np.sqrt(cv_mse_scores)
    
    logger.info(f"Cross-validation R² scores: {cv_r2_scores}")
    logger.info(f"Mean CV R²: {cv_r2_scores.mean():.4f} (+/- {cv_r2_scores.std() * 2:.4f})")
    logger.info(f"Mean CV RMSE: {cv_rmse_scores.mean():.4f} (+/- {cv_rmse_scores.std() * 2:.4f})")

    # Train final model on full training set
    if is_default:
        # Use early stopping for default model
        xgb_model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
            verbose=False
        )
    else:
        xgb_model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
            verbose=False
        )
    
    # Get training history
    results = xgb_model.evals_result()
    train_loss = results['validation_0']['rmse']
    test_loss = results['validation_1']['rmse']
    
    xgb_preds = xgb_model.predict(X_test_scaled)
    xgb_mse = mean_squared_error(y_test, xgb_preds)
    xgb_rmse = np.sqrt(xgb_mse)
    xgb_r2 = r2_score(y_test, xgb_preds)
    logger.info("Final Test Set Performance:")
    logger.info(f"XGBoost Test MSE: {xgb_mse:.4f}")
    logger.info(f"XGBoost Test RMSE: {xgb_rmse:.4f}")
    logger.info(f"XGBoost Test R-squared: {xgb_r2:.4f}")

    # Plot Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, xgb_preds, alpha=0.5, color='orange')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel("Actual Target")
    plt.ylabel("Predicted Target")
    plt.title("XGBoost Actual vs Predicted")
    plot_path = os.path.join(models_dir, f"actual_vs_predicted_{timestamp}.png")
    plt.savefig(plot_path)
    logger.info(f"Actual vs Predicted plot saved to {plot_path}")
    plt.close()

    # --- Save model and scaler ---
    logger.info("Saving model and scaler...")
    joblib.dump(xgb_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Save the training data
    training_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'target_column': target_column
    }
    joblib.dump(training_data, training_data_path)
    
    logger.info(f"XGBoost model saved to {model_path}")
    logger.info(f"Scaler saved to {scaler_path}")
    logger.info(f"Training data saved to {training_data_path}")
    
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
                'rmse_std': float(cv_rmse_scores.std())
            },
            'loss_curve': {
                'train_loss': [float(x) for x in train_loss],
                'test_loss': [float(x) for x in test_loss]
            }
        },
        'model_path': model_path,
        'training_data_path': training_data_path,
        'actual': y_test.tolist(),
        'predicted': xgb_preds.tolist()
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Model metrics saved to {metrics_path}")
    
    return metrics

def train_nextday_model(data_source, models_dir, target_column=None, datetime_col=None, start_date=None, freq=None, window_hours=24, test_size=0.2):
    """
    Trains a next-day prediction model using sliding windows and engineered features.
    
    Args:
        data_source: Path to the data file
        models_dir: Directory to save model files
        target_column: Name of the target column
        datetime_col: Name of the datetime column
        start_date: Start date if no datetime column
        freq: Frequency of the data
        window_hours: Number of hours to use as input window
        test_size: Fraction of data to use for testing
    
    Returns:
        Dictionary containing training metrics
    """
    logger.info("Starting next-day prediction model training...")
    
    # Load and parse data
    if datetime_col is None and start_date is None:
        raise ValueError("Either 'datetime_col' or 'start_date' must be provided.")
    
    df = load_and_parse(data_source, datetime_col=datetime_col, start_date=start_date, freq=freq, silent=False)
    
    # Ensure hourly frequency
    if df.index.freq != 'H' and pd.infer_freq(df.index) != 'H':
        logger.info("Resampling data to hourly frequency...")
        df = df.resample('H').mean().interpolate()
    
    # If target_column is not specified, use the last column
    if target_column is None:
        target_column = df.columns[-1]
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")
    
    # --- Feature Engineering ---
    # Drop the specified column if it exists
    pressure_col = 'Heat Exchanger Pressure Differential (Bar)'
    if pressure_col in df.columns:
        df = df.drop(columns=[pressure_col])
        logger.info(f"Dropped column: {pressure_col}")

    # Create next-day features
    logger.info("Creating next-day prediction features...")
    X, y = create_nextday_features(df, target_column, window_hours, silent=False)
    
    if len(X) == 0:
        raise ValueError("No samples could be created. Check data length and window size.")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    logger.info(f"Training data shape: {X_train_scaled.shape}")
    logger.info(f"Test data shape: {X_test_scaled.shape}")
    
    # Train multiple models (one for each hour of the next day)
    models = []
    predictions = []
    metrics_per_hour = []
    
    logger.info("Training 24 models (one for each hour of the next day)...")
    
    for hour in range(24):
        logger.info(f"Training model for hour {hour}...")
        
        # Get target for this hour
        y_train_hour = y_train.iloc[:, hour]
        y_test_hour = y_test.iloc[:, hour]
        
        # Train XGBoost model for this hour
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='rmse',
            early_stopping_rounds=30
        )
        
        # Train with early stopping
        model.fit(
            X_train_scaled, y_train_hour,
            eval_set=[(X_train_scaled, y_train_hour), (X_test_scaled, y_test_hour)],
            verbose=False
        )
        
        # Make predictions
        y_pred_hour = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_hour, y_pred_hour)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_hour, y_pred_hour)
        
        models.append(model)
        predictions.append(y_pred_hour)
        metrics_per_hour.append({
            'hour': hour,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        })
        
        logger.info(f"Hour {hour}: RMSE={rmse:.4f}, R²={r2:.4f}")
    
    # Calculate overall metrics
    y_test_flat = y_test.values.flatten()
    y_pred_flat = np.array(predictions).T.flatten()
    
    overall_mse = mean_squared_error(y_test_flat, y_pred_flat)
    overall_rmse = np.sqrt(overall_mse)
    overall_r2 = r2_score(y_test_flat, y_pred_flat)
    
    logger.info(f"Overall performance: RMSE={overall_rmse:.4f}, R²={overall_r2:.4f}")

    # --- Find and save example predictions ---
    y_pred_test = np.array(predictions).T  # Transpose to have shape (n_samples, 24)
    
    example_results = []
    for i in range(len(X_test)):
        actuals = y_test.iloc[i].values
        preds = y_pred_test[i]
        
        # Ensure we have a full 24-hour forecast to evaluate
        if len(actuals) == 24 and len(preds) == 24:
            rmse = np.sqrt(mean_squared_error(actuals, preds))
            example_results.append({
                'input_features': X_test.iloc[i].to_dict(),
                'actual_values': actuals.tolist(),
                'predicted_values': preds.tolist(),
                'rmse': rmse,
                'input_timestamp': X_test.index[i]
            })

    if example_results:
        # Sort examples by RMSE to find the median
        example_results.sort(key=lambda x: x['rmse'])
        
        # Get 5 examples centered around the median RMSE
        num_examples = len(example_results)
        if num_examples >= 5:
            median_index = num_examples // 2
            start_index = max(0, median_index - 2)
            end_index = start_index + 5
            # Adjust if we're at the end of the list
            if end_index > num_examples:
                end_index = num_examples
                start_index = end_index - 5
            
            examples_to_save = example_results[start_index:end_index]
            logger.info(f"Saving 5 average prediction examples (centered around median RMSE) from the test set.")
        else:
            # If less than 5 examples, just save all of them
            examples_to_save = example_results
            logger.info(f"Saving all {num_examples} prediction examples from the test set (fewer than 5 available).")
        
        # Save the examples list
        examples_path = os.path.join(models_dir, "nextday_examples.pkl")
        joblib.dump(examples_to_save, examples_path)
        logger.info(f"Test examples saved to {examples_path}")

        # Remove the old single best example file if it exists
        old_example_path = os.path.join(models_dir, "nextday_best_example.pkl")
        if os.path.exists(old_example_path):
            try:
                os.remove(old_example_path)
                logger.info(f"Removed old example file: {old_example_path}")
            except OSError as e:
                logger.error(f"Error removing old example file {old_example_path}: {e}")

    # Save test data for UI components
    test_data_path = os.path.join(models_dir, "nextday_test_data.pkl")
    joblib.dump((X_test, y_test), test_data_path)
    logger.info(f"Test data saved for UI: {test_data_path}")
    
    # Save models and scaler
    model_path = os.path.join(models_dir, "nextday_model.pkl")
    scaler_path = os.path.join(models_dir, "nextday_scaler.pkl")
    feature_names_path = os.path.join(models_dir, "nextday_feature_names.pkl")
    metrics_path = os.path.join(models_dir, "nextday_metrics.json")
    
    joblib.dump(models, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(X.columns.tolist(), feature_names_path)
    
    # Save metrics
    metrics = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'model_type': 'NextDay_XGBoost',
        'window_hours': window_hours,
        'overall_metrics': {
            'mse': float(overall_mse),
            'rmse': float(overall_rmse),
            'r2': float(overall_r2)
        },
        'hourly_metrics': metrics_per_hour,
        'model_path': model_path,
        'scaler_path': scaler_path,
        'feature_names_path': feature_names_path
    }

    if example_results:
        # Save RMSEs for the top examples for reference
        metrics['example_rmses'] = [ex['rmse'] for ex in examples_to_save]
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Next-day prediction model saved to {model_path}")
    logger.info(f"Scaler saved to {scaler_path}")
    logger.info(f"Metrics saved to {metrics_path}")
    
    return metrics

if __name__ == "__main__":
    pass
    # DEPRECATED
