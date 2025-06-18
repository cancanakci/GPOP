import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os
from datetime import datetime

def load_and_split_data():
    """Load the full dataset and split it temporally into 2020-2022 (train) and 2023 (test)."""
    data_path = "trial_grounds/data/raw/default_data_nopressure.xlsx"
    
    try:
        data = pd.read_excel(data_path)
        print(f"Loaded full dataset: {data.shape}")
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
            
            # Check date range
            print(f"Date range: {data[datetime_col].min()} to {data[datetime_col].max()}")
        else:
            print("No datetime column found!")
            return None, None, None
        
        # Drop pressure column if present
        if 'Heat Exchanger Pressure Differential (Bar)' in data.columns:
            data = data.drop(columns=['Heat Exchanger Pressure Differential (Bar)'])
            print("Dropped pressure column")
        
        # Handle missing values
        print(f"\nMissing values before handling:")
        print(data.isnull().sum())
        data = data.ffill().bfill()
        
        # Split temporally: 2020-2022 for training, 2023 for testing
        train_data = data[data[datetime_col].dt.year <= 2022].copy()
        test_data = data[data[datetime_col].dt.year == 2023].copy()
        
        print(f"\nTemporal split:")
        print(f"Training data (2020-2022): {train_data.shape}")
        print(f"Test data (2023): {test_data.shape}")
        
        if len(train_data) == 0 or len(test_data) == 0:
            print("Error: No data in one of the splits!")
            return None, None, None
        
        return train_data, test_data, datetime_col
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def prepare_features_and_target(data, target_column='Gross Power (MW)'):
    """Prepare features and target from the data."""
    feature_columns = [
        'Brine Flowrate (T/h)',
        'Fluid Temperature (°C)',
        'NCG+Steam Flowrate (T/h)',
        'Ambient Temperature (°C)',
        'Reinjection Temperature (°C)'
    ]
    
    # Check if all required columns are present
    missing_cols = set(feature_columns + [target_column]) - set(data.columns)
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return None, None
    
    X = data[feature_columns]
    y = data[target_column]
    
    return X, y

def train_model(X_train, y_train, X_test, y_test):
    """Train XGBoost model on 2020-2022 data."""
    print("\n=== Training XGBoost Model ===")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
        verbose=False
    )
    
    # Make predictions
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_metrics = {
        'mse': mean_squared_error(y_train, train_preds),
        'rmse': np.sqrt(mean_squared_error(y_train, train_preds)),
        'mae': mean_absolute_error(y_train, train_preds),
        'r2': r2_score(y_train, train_preds)
    }
    
    test_metrics = {
        'mse': mean_squared_error(y_test, test_preds),
        'rmse': np.sqrt(mean_squared_error(y_test, test_preds)),
        'mae': mean_absolute_error(y_test, test_preds),
        'r2': r2_score(y_test, test_preds),
        'mape': np.mean(np.abs((y_test - test_preds) / y_test)) * 100
    }
    
    print(f"\nTraining Set Performance (2020-2022):")
    print(f"R²: {train_metrics['r2']:.4f}")
    print(f"RMSE: {train_metrics['rmse']:.4f}")
    print(f"MAE: {train_metrics['mae']:.4f}")
    
    print(f"\nTest Set Performance (2023):")
    print(f"R²: {test_metrics['r2']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"MAPE: {test_metrics['mape']:.2f}%")
    
    return model, scaler, train_preds, test_preds, train_metrics, test_metrics

def create_plots(train_data, test_data, train_preds, test_preds, datetime_col, train_metrics, test_metrics):
    """Create comprehensive evaluation plots."""
    output_dir = "trial_grounds/results/temporal_split_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Predictions vs Actuals for both train and test
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Training Set (2020-2022)', 'Test Set (2023)']
    )
    
    # Training set
    fig.add_trace(
        go.Scatter(
            x=train_data['Gross Power (MW)'],
            y=train_preds,
            mode='markers',
            name='Training Predictions',
            marker=dict(color='blue', opacity=0.6)
        ),
        row=1, col=1
    )
    
    # Test set
    fig.add_trace(
        go.Scatter(
            x=test_data['Gross Power (MW)'],
            y=test_preds,
            mode='markers',
            name='Test Predictions',
            marker=dict(color='red', opacity=0.6)
        ),
        row=1, col=2
    )
    
    # Add perfect prediction lines
    for col in [1, 2]:
        min_val = min(train_data['Gross Power (MW)'].min(), test_data['Gross Power (MW)'].min())
        max_val = max(train_data['Gross Power (MW)'].max(), test_data['Gross Power (MW)'].max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='black', dash='dash'),
                showlegend=False
            ),
            row=1, col=col
        )
    
    fig.update_layout(
        title='Temporal Split: Predictions vs Actuals',
        height=500
    )
    
    fig.write_html(os.path.join(output_dir, 'predictions_vs_actuals.html'))
    
    # 2. Time Series Plot
    fig_ts = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Time Series: Actual vs Predicted', 'Prediction Errors Over Time'],
        vertical_spacing=0.1
    )
    
    # Training data time series
    fig_ts.add_trace(
        go.Scatter(
            x=train_data[datetime_col],
            y=train_data['Gross Power (MW)'],
            mode='lines',
            name='Training Actual',
            line=dict(color='blue'),
            opacity=0.7
        ),
        row=1, col=1
    )
    
    fig_ts.add_trace(
        go.Scatter(
            x=train_data[datetime_col],
            y=train_preds,
            mode='lines',
            name='Training Predicted',
            line=dict(color='lightblue', dash='dash'),
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Test data time series
    fig_ts.add_trace(
        go.Scatter(
            x=test_data[datetime_col],
            y=test_data['Gross Power (MW)'],
            mode='lines',
            name='Test Actual',
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    fig_ts.add_trace(
        go.Scatter(
            x=test_data[datetime_col],
            y=test_preds,
            mode='lines',
            name='Test Predicted',
            line=dict(color='orange', dash='dash')
        ),
        row=1, col=1
    )
    
    # Training errors
    train_errors = train_data['Gross Power (MW)'] - train_preds
    fig_ts.add_trace(
        go.Scatter(
            x=train_data[datetime_col],
            y=train_errors,
            mode='lines',
            name='Training Errors',
            line=dict(color='blue'),
            opacity=0.5
        ),
        row=2, col=1
    )
    
    # Test errors
    test_errors = test_data['Gross Power (MW)'] - test_preds
    fig_ts.add_trace(
        go.Scatter(
            x=test_data[datetime_col],
            y=test_errors,
            mode='lines',
            name='Test Errors',
            line=dict(color='red')
        ),
        row=2, col=1
    )
    
    # Zero line
    all_dates = pd.concat([train_data[datetime_col], test_data[datetime_col]])
    fig_ts.add_trace(
        go.Scatter(
            x=all_dates,
            y=[0] * len(all_dates),
            mode='lines',
            name='Zero Error',
            line=dict(color='black', dash='dash')
        ),
        row=2, col=1
    )
    
    fig_ts.update_layout(
        title='Temporal Split: Time Series Analysis',
        height=800
    )
    
    fig_ts.update_yaxes(title_text="Gross Power (MW)", row=1, col=1)
    fig_ts.update_yaxes(title_text="Prediction Error (MW)", row=2, col=1)
    
    fig_ts.write_html(os.path.join(output_dir, 'time_series_analysis.html'))
    
    # 3. Performance Comparison
    fig_comp = go.Figure()
    
    metrics = ['R²', 'RMSE', 'MAE']
    train_values = [train_metrics['r2'], train_metrics['rmse'], train_metrics['mae']]
    test_values = [test_metrics['r2'], test_metrics['rmse'], test_metrics['mae']]
    
    fig_comp.add_trace(go.Bar(
        x=metrics,
        y=train_values,
        name='Training (2020-2022)',
        marker_color='blue'
    ))
    
    fig_comp.add_trace(go.Bar(
        x=metrics,
        y=test_values,
        name='Test (2023)',
        marker_color='red'
    ))
    
    fig_comp.update_layout(
        title='Performance Comparison: Training vs Test',
        yaxis_title='Metric Value',
        barmode='group'
    )
    
    fig_comp.write_html(os.path.join(output_dir, 'performance_comparison.html'))
    
    # 4. Save results
    results_df = pd.DataFrame({
        'Dataset': ['Training (2020-2022)', 'Test (2023)'],
        'R²': [train_metrics['r2'], test_metrics['r2']],
        'RMSE': [train_metrics['rmse'], test_metrics['rmse']],
        'MAE': [train_metrics['mae'], test_metrics['mae']],
        'MAPE': [None, test_metrics['mape']]
    })
    
    results_df.to_csv(os.path.join(output_dir, 'performance_results.csv'), index=False)
    
    print(f"\nAll plots saved to {output_dir}")

def main():
    """Main function for temporal split evaluation."""
    print("=== Temporal Split Evaluation: 2020-2022 Train, 2023 Test ===")
    
    # Load and split data
    train_data, test_data, datetime_col = load_and_split_data()
    if train_data is None:
        return
    
    # Prepare features and target
    X_train, y_train = prepare_features_and_target(train_data)
    X_test, y_test = prepare_features_and_target(test_data)
    
    if X_train is None or X_test is None:
        return
    
    # Train model
    model, scaler, train_preds, test_preds, train_metrics, test_metrics = train_model(
        X_train, y_train, X_test, y_test
    )
    
    # Create plots
    create_plots(train_data, test_data, train_preds, test_preds, datetime_col, train_metrics, test_metrics)
    
    # Save model
    output_dir = "trial_grounds/results/temporal_split_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(output_dir, 'temporal_model.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'temporal_scaler.pkl'))
    joblib.dump(X_train.columns.tolist(), os.path.join(output_dir, 'temporal_feature_names.pkl'))
    
    print(f"\nModel saved to {output_dir}")
    print("\n=== Evaluation Complete ===")

if __name__ == "__main__":
    main() 