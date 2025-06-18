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
from datetime import datetime, timedelta

class TemporalDriftAnalyzer:
    def __init__(self, data_path="../../data/default_data.xlsx"):
        """Initialize the temporal drift analyzer."""
        self.data_path = data_path
        self.output_dir = "../plots"
        os.makedirs(self.output_dir, exist_ok=True)
        self.train_years = [2020, 2021, 2022, 2023]
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the full dataset."""
        print("Loading dataset...")
        data = pd.read_excel(self.data_path)
        print(f"Dataset shape: {data.shape}")
        
        # Handle datetime
        datetime_col = 'Datetime'
        data[datetime_col] = pd.to_datetime(data[datetime_col])
        print(f"Date range: {data[datetime_col].min()} to {data[datetime_col].max()}")
        
        # Drop pressure column if present
        if 'Heat Exchanger Pressure Differential (Bar)' in data.columns:
            data = data.drop(columns=['Heat Exchanger Pressure Differential (Bar)'])
        
        # Handle missing values
        data = data.ffill().bfill()
        
        return data, datetime_col
    
    def add_temporal_features(self, data, datetime_col, train_end_date=None):
        """Add temporal features to the dataset."""
        data = data.copy()
        
        # Basic temporal features
        data['hour'] = data[datetime_col].dt.hour
        data['day_of_week'] = data[datetime_col].dt.dayofweek
        data['month'] = data[datetime_col].dt.month
        data['day_of_year'] = data[datetime_col].dt.dayofyear
        data['year'] = data[datetime_col].dt.year
        
        # Time since last training row (if train_end_date provided)
        if train_end_date is not None:
            data['days_since_training'] = (data[datetime_col] - train_end_date).dt.days
            data['hours_since_training'] = (data[datetime_col] - train_end_date).dt.total_seconds() / 3600
        
        # Rolling statistics (last 24 hours)
        for col in ['Brine Flowrate (T/h)', 'Fluid Temperature (°C)', 'NCG+Steam Flowrate (T/h)']:
            data[f'{col}_24h_mean'] = data[col].rolling(window=24, min_periods=1).mean()
            data[f'{col}_24h_std'] = data[col].rolling(window=24, min_periods=1).std()
        
        return data
    
    def split_data_by_years(self, data, datetime_col, train_years, test_years):
        """Split data by specified years."""
        train_data = data[data[datetime_col].dt.year.isin(train_years)].copy()
        test_data = data[data[datetime_col].dt.year.isin(test_years)].copy()
        
        print(f"Training data ({train_years}): {train_data.shape}")
        print(f"Test data ({test_years}): {test_data.shape}")
        
        return train_data, test_data
    
    def prepare_features(self, data, use_temporal_features=True):
        """Prepare features for training/prediction."""
        base_features = [
            'Brine Flowrate (T/h)',
            'Fluid Temperature (°C)',
            'NCG+Steam Flowrate (T/h)',
            'Ambient Temperature (°C)',
            'Reinjection Temperature (°C)'
        ]
        
        if use_temporal_features:
            temporal_features = [
                'hour', 'day_of_week', 'month', 'day_of_year', 'year',
                'days_since_training', 'hours_since_training',
                'Brine Flowrate (T/h)_24h_mean', 'Brine Flowrate (T/h)_24h_std',
                'Fluid Temperature (°C)_24h_mean', 'Fluid Temperature (°C)_24h_std',
                'NCG+Steam Flowrate (T/h)_24h_mean', 'NCG+Steam Flowrate (T/h)_24h_std'
            ]
            feature_cols = base_features + temporal_features
        else:
            feature_cols = base_features
        
        # Filter to available columns
        available_features = [col for col in feature_cols if col in data.columns]
        print(f"Using {len(available_features)} features: {available_features}")
        
        X = data[available_features]
        y = data['Gross Power (MW)']
        
        return X, y, available_features
    
    def train_and_predict(self, train_data, full_data):
        X_train, y_train, _ = self.prepare_features(train_data)
        X_full, y_full, _ = self.prepare_features(full_data)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_full_scaled = scaler.transform(X_full)
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
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_full_scaled)
        return preds, y_full
    
    def run(self):
        data, datetime_col = self.load_and_prepare_data()
        all_errors = {}
        for year in self.train_years:
            print(f"Training on {year}, predicting on 2020-2023...")
            train_data = data[data[datetime_col].dt.year == year]
            preds, y_full = self.train_and_predict(train_data, data)
            errors = y_full.values - preds
            all_errors[year] = errors
            # Mask for training period
            is_train_period = data[datetime_col].dt.year == year
            # Plot: blue for training period, red for rest
            fig = go.Figure()
            # Blue: training period
            fig.add_trace(go.Scatter(
                x=data[datetime_col][is_train_period],
                y=errors[is_train_period],
                mode='lines',
                name=f'Error (Train {year})',
                line=dict(color='blue'),
                opacity=0.8
            ))
            # Red: out-of-training period
            fig.add_trace(go.Scatter(
                x=data[datetime_col][~is_train_period],
                y=errors[~is_train_period],
                mode='lines',
                name=f'Error (Test)',
                line=dict(color='red'),
                opacity=0.8
            ))
            # Zero error line
            fig.add_trace(go.Scatter(
                x=data[datetime_col],
                y=[0]*len(data),
                mode='lines',
                name='Zero Error',
                line=dict(color='black', dash='dash'),
                showlegend=True
            ))
            fig.update_layout(
                title=f'Prediction Error Over Time (Trained on {year})',
                xaxis_title='Datetime',
                yaxis_title='Error (Actual - Predicted Gross Power MW)',
                legend_title='Error Period',
                height=600
            )
            fig.write_html(os.path.join(self.output_dir, f'temporal_drift_error_trained_on_{year}.html'))
            print(f"Plot saved to {self.output_dir}/temporal_drift_error_trained_on_{year}.html")
        # Combined plot (unchanged)
        fig_all = go.Figure()
        for year in self.train_years:
            fig_all.add_trace(go.Scatter(
                x=data[datetime_col],
                y=all_errors[year],
                mode='lines',
                name=f'Trained on {year}',
                opacity=0.7
            ))
        fig_all.add_trace(go.Scatter(
            x=data[datetime_col],
            y=[0]*len(data),
            mode='lines',
            name='Zero Error',
            line=dict(color='black', dash='dash'),
            showlegend=True
        ))
        fig_all.update_layout(
            title='Prediction Error Over Time by Training Year',
            xaxis_title='Datetime',
            yaxis_title='Error (Actual - Predicted Gross Power MW)',
            legend_title='Model Training Year',
            height=600
        )
        fig_all.write_html(os.path.join(self.output_dir, 'temporal_drift_error_by_training_year.html'))
        print(f"Combined plot saved to {self.output_dir}/temporal_drift_error_by_training_year.html")

def main():
    """Main function to run temporal drift analysis."""
    analyzer = TemporalDriftAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main() 