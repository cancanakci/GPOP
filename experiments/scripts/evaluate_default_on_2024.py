import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def load_default_model():
    """Load the default model, scaler, and feature names."""
    models_dir = "models"
    
    try:
        model = joblib.load(os.path.join(models_dir, "default_model.pkl"))
        scaler = joblib.load(os.path.join(models_dir, "default_scaler.pkl"))
        feature_names = joblib.load(os.path.join(models_dir, "default_feature_names.pkl"))
        
        print("Default model loaded successfully!")
        print(f"Feature names: {feature_names}")
        return model, scaler, feature_names
    except Exception as e:
        print(f"Error loading default model: {e}")
        return None, None, None

def load_and_prepare_2024_data():
    """Load and prepare the daily_2024 data."""
    data_path = "trial_grounds/data/raw/daily_2024.xlsx"
    
    try:
        data = pd.read_excel(data_path)
        print(f"Loaded daily_2024 data: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        
        # Handle datetime column
        datetime_col = None
        for col in ['Datetime', 'Timestamp', 'Date']:
            if col in data.columns:
                datetime_col = col
                break
        
        if datetime_col:
            data[datetime_col] = pd.to_datetime(data[datetime_col])
            print(f"Using datetime column: {datetime_col}")
        
        # Check for missing values
        print(f"\nMissing values:")
        print(data.isnull().sum())
        
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        return data, datetime_col
    except Exception as e:
        print(f"Error loading daily_2024 data: {e}")
        return None, None

def prepare_features(data, feature_names):
    """Prepare features for prediction."""
    # Check if all required features are present
    missing_features = set(feature_names) - set(data.columns)
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        return None
    
    # Select only the required features
    X = data[feature_names]
    print(f"Prepared features shape: {X.shape}")
    
    return X

def make_predictions(model, scaler, X, data, datetime_col):
    """Make predictions and calculate metrics."""
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    # Get actual values
    target_col = 'Gross Power (MW)'
    if target_col not in data.columns:
        print(f"Error: Target column '{target_col}' not found in data")
        return None
    
    actuals = data[target_col].values
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    print(f"\n=== Performance Metrics ===")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    return predictions, actuals, datetime_col

def create_plots(predictions, actuals, datetime_col, data):
    """Create prediction vs actual plots and time series plots."""
    # Create output directory
    output_dir = "trial_grounds/results/default_model_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Predictions vs Actuals Scatter Plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=actuals, 
        y=predictions, 
        mode='markers',
        name='Predictions vs Actuals',
        marker=dict(color='blue', opacity=0.6)
    ))
    
    # Add perfect prediction line
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], 
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='2020-2023 Model: Predictions vs Actuals (Daily 2024)',
        xaxis_title='Actual Gross Power (MW)',
        yaxis_title='Predicted Gross Power (MW)',
        width=800,
        height=600
    )
    
    fig.write_html(os.path.join(output_dir, 'predictions_vs_actuals.html'))
    print(f"Predictions vs Actuals plot saved to {output_dir}")
    
    # 2. Time Series Plot
    if datetime_col:
        fig_ts = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Time Series: Actual vs Predicted', 'Prediction Errors Over Time'],
            vertical_spacing=0.1
        )
        
        # Time series of actual vs predicted
        fig_ts.add_trace(
            go.Scatter(
                x=data[datetime_col],
                y=actuals,
                mode='lines+markers',
                name='Actual',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        fig_ts.add_trace(
            go.Scatter(
                x=data[datetime_col],
                y=predictions,
                mode='lines+markers',
                name='Predicted',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # Prediction errors over time
        errors = actuals - predictions
        fig_ts.add_trace(
            go.Scatter(
                x=data[datetime_col],
                y=errors,
                mode='lines+markers',
                name='Prediction Error',
                line=dict(color='orange')
            ),
            row=2, col=1
        )
        
        # Add zero line for errors
        fig_ts.add_trace(
            go.Scatter(
                x=data[datetime_col],
                y=[0] * len(errors),
                mode='lines',
                name='Zero Error',
                line=dict(color='black', dash='dash')
            ),
            row=2, col=1
        )
        
        fig_ts.update_layout(
            title='2020-2023 Model: Time Series Analysis (Daily 2024)',
            height=800,
            showlegend=True
        )
        
        fig_ts.update_yaxes(title_text="Gross Power (MW)", row=1, col=1)
        fig_ts.update_yaxes(title_text="Prediction Error (MW)", row=2, col=1)
        fig_ts.update_xaxes(title_text="Time", row=2, col=1)
        
        fig_ts.write_html(os.path.join(output_dir, 'time_series_analysis.html'))
        print(f"Time series analysis plot saved to {output_dir}")
    
    # 3. Error Distribution
    errors = actuals - predictions
    fig_error = go.Figure()
    
    fig_error.add_trace(go.Histogram(
        x=errors,
        nbinsx=20,
        name='Error Distribution'
    ))
    
    fig_error.update_layout(
        title='2020-2023 Model: Prediction Error Distribution (Daily 2024)',
        xaxis_title='Prediction Error (MW)',
        yaxis_title='Count',
        width=800,
        height=500
    )
    
    fig_error.write_html(os.path.join(output_dir, 'error_distribution.html'))
    print(f"Error distribution plot saved to {output_dir}")
    
    # 4. Save results to CSV
    results_df = pd.DataFrame({
        'Timestamp': data[datetime_col] if datetime_col else range(len(actuals)),
        'Actual': actuals,
        'Predicted': predictions,
        'Error': errors,
        'Absolute_Error': np.abs(errors)
    })
    
    results_df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
    print(f"Results saved to {output_dir}/evaluation_results.csv")

def main():
    """Main function to evaluate default model on daily_2024."""
    print("=== Evaluating Default Model on Daily 2024 Data ===")
    
    # Load default model
    model, scaler, feature_names = load_default_model()
    if model is None:
        return
    
    # Load daily_2024 data
    data, datetime_col = load_and_prepare_2024_data()
    if data is None:
        return
    
    # Prepare features
    X = prepare_features(data, feature_names)
    if X is None:
        return
    
    # Make predictions
    results = make_predictions(model, scaler, X, data, datetime_col)
    if results is None:
        return
    
    predictions, actuals, datetime_col = results
    
    # Create plots
    create_plots(predictions, actuals, datetime_col, data)
    
    print(f"\n=== Evaluation Complete ===")
    print("Check the plots in trial_grounds/results/default_model_evaluation/")

if __name__ == "__main__":
    main() 