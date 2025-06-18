import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os

class TrainTestSplitsAnalyzer:
    def __init__(self, data_path="experiments/data/default_data.xlsx"):
        self.data_path = data_path
        self.output_dir = "experiments/plots"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define the 4 planned train-test splits
        self.splits = [
            {
                'name': 'Train_2020_Test_2021-2022-2023',
                'train_years': [2020],
                'test_years': [2021, 2022, 2023],
                'description': 'Train on 2020, test on 2021-2022-2023'
            },
            {
                'name': 'Train_2021_Test_2020-2022-2023',
                'train_years': [2021],
                'test_years': [2020, 2022, 2023],
                'description': 'Train on 2021, test on 2020-2022-2023'
            },
            {
                'name': 'Train_2022_Test_2020-2021-2023',
                'train_years': [2022],
                'test_years': [2020, 2021, 2023],
                'description': 'Train on 2022, test on 2020-2021-2023'
            },
            {
                'name': 'Train_2023_Test_2020-2021-2022',
                'train_years': [2023],
                'test_years': [2020, 2021, 2022],
                'description': 'Train on 2023, test on 2020-2021-2022'
            }
        ]
        
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset."""
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
    
    def prepare_features(self, data):
        """Prepare features for training/prediction."""
        features = [
            'Brine Flowrate (T/h)',
            'Fluid Temperature (°C)',
            'NCG+Steam Flowrate (T/h)',
            'Ambient Temperature (°C)',
            'Reinjection Temperature (°C)'
        ]
        
        X = data[features]
        y = data['Gross Power (MW)']
        
        return X, y, features
    
    def split_data_by_years(self, data, datetime_col, train_years, test_years):
        """Split data by specified years."""
        train_data = data[data[datetime_col].dt.year.isin(train_years)].copy()
        test_data = data[data[datetime_col].dt.year.isin(test_years)].copy()
        
        print(f"Training data ({train_years}): {train_data.shape}")
        print(f"Test data ({test_years}): {test_data.shape}")
        
        return train_data, test_data
    
    def train_and_evaluate(self, train_data, test_data, datetime_col):
        """Train model and evaluate on test data."""
        # Prepare features
        X_train, y_train, feature_names = self.prepare_features(train_data)
        X_test, y_test, _ = self.prepare_features(test_data)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
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
        
        # Make predictions
        train_preds = model.predict(X_train_scaled)
        test_preds = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_metrics = {
            'r2': r2_score(y_train, train_preds),
            'rmse': np.sqrt(mean_squared_error(y_train, train_preds)),
            'mae': mean_absolute_error(y_train, train_preds)
        }
        
        test_metrics = {
            'r2': r2_score(y_test, test_preds),
            'rmse': np.sqrt(mean_squared_error(y_test, test_preds)),
            'mae': mean_absolute_error(y_test, test_preds),
            'mape': np.mean(np.abs((y_test - test_preds) / y_test)) * 100
        }
        
        return {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'train_preds': train_preds,
            'test_preds': test_preds,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_data': train_data,
            'test_data': test_data
        }
    
    def create_split_visualization(self, split_config, results, data, datetime_col):
        """Create comprehensive visualization for a single train-test split."""
        split_name = split_config['name']
        description = split_config['description']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f'Actual vs Predicted (Train)',
                f'Actual vs Predicted (Test)',
                f'Error Over Time (Test)',
                f'Error Distribution (Test)'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Actual vs Predicted - Training
        train_data = results['train_data']
        train_preds = results['train_preds']
        
        fig.add_trace(
            go.Scatter(
                x=train_data['Gross Power (MW)'],
                y=train_preds,
                mode='markers',
                name='Train Predictions',
                marker=dict(color='blue', opacity=0.6, size=3)
            ),
            row=1, col=1
        )
        
        # Add perfect prediction line
        min_val = min(train_data['Gross Power (MW)'].min(), train_preds.min())
        max_val = max(train_data['Gross Power (MW)'].max(), train_preds.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Actual vs Predicted - Test
        test_data = results['test_data']
        test_preds = results['test_preds']
        
        fig.add_trace(
            go.Scatter(
                x=test_data['Gross Power (MW)'],
                y=test_preds,
                mode='markers',
                name='Test Predictions',
                marker=dict(color='green', opacity=0.6, size=3)
            ),
            row=1, col=2
        )
        
        # Add perfect prediction line
        min_val = min(test_data['Gross Power (MW)'].min(), test_preds.min())
        max_val = max(test_data['Gross Power (MW)'].max(), test_preds.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Error over time
        test_errors = test_data['Gross Power (MW)'] - test_preds
        
        fig.add_trace(
            go.Scatter(
                x=test_data[datetime_col],
                y=test_errors,
                mode='lines',
                name='Test Errors',
                line=dict(color='purple'),
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Add zero line
        fig.add_trace(
            go.Scatter(
                x=test_data[datetime_col],
                y=[0] * len(test_errors),
                mode='lines',
                name='Zero Error',
                line=dict(color='black', dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Error distribution
        fig.add_trace(
            go.Histogram(
                x=test_errors,
                name='Error Distribution',
                nbinsx=50,
                marker_color='orange',
                opacity=0.7
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Train-Test Split Analysis: {description}',
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Actual Power (MW)", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Power (MW)", row=1, col=1)
        fig.update_xaxes(title_text="Actual Power (MW)", row=1, col=2)
        fig.update_yaxes(title_text="Predicted Power (MW)", row=1, col=2)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Error (Actual - Predicted)", row=2, col=1)
        fig.update_xaxes(title_text="Error (Actual - Predicted)", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        # Save plot
        filename = f"train_test_split_{split_name.lower().replace('_', '-')}.html"
        fig.write_html(os.path.join(self.output_dir, filename))
        print(f"Plot saved to {self.output_dir}/{filename}")
        
        return fig
    
    def run_single_split(self, split_config, data, datetime_col):
        """Run analysis for a single train-test split."""
        print(f"\n=== Running Split: {split_config['name']} ===")
        print(f"Description: {split_config['description']}")
        
        # Split data
        train_data, test_data = self.split_data_by_years(
            data, datetime_col, split_config['train_years'], split_config['test_years']
        )
        
        if len(train_data) == 0 or len(test_data) == 0:
            print("Warning: Empty train or test set!")
            return None
        
        # Train and evaluate
        results = self.train_and_evaluate(train_data, test_data, datetime_col)
        
        # Print results
        print(f"\nTraining Metrics:")
        print(f"  R²: {results['train_metrics']['r2']:.4f}")
        print(f"  RMSE: {results['train_metrics']['rmse']:.4f}")
        print(f"  MAE: {results['train_metrics']['mae']:.4f}")
        
        print(f"\nTest Metrics:")
        print(f"  R²: {results['test_metrics']['r2']:.4f}")
        print(f"  RMSE: {results['test_metrics']['rmse']:.4f}")
        print(f"  MAE: {results['test_metrics']['mae']:.4f}")
        print(f"  MAPE: {results['test_metrics']['mape']:.2f}%")
        
        # Create visualization
        self.create_split_visualization(split_config, results, data, datetime_col)
        
        return results
    
    def run_all_splits(self):
        """Run analysis for all train-test splits."""
        print("=== Train-Test Splits Analysis ===")
        
        # Load data
        data, datetime_col = self.load_and_prepare_data()
        
        # Run each split
        for split_config in self.splits:
            results = self.run_single_split(split_config, data, datetime_col)
            if results:
                self.results[split_config['name']] = results
        
        # Create summary comparison
        self.create_summary_comparison()
        
        print(f"\n=== Analysis Complete ===")
        print(f"Results saved to {self.output_dir}")
    
    def create_summary_comparison(self):
        """Create a summary comparison of all splits."""
        if not self.results:
            return
        
        # Prepare summary data
        summary_data = []
        for split_name, results in self.results.items():
            summary_data.append({
                'Split': split_name,
                'Train_R2': results['train_metrics']['r2'],
                'Test_R2': results['test_metrics']['r2'],
                'Train_RMSE': results['train_metrics']['rmse'],
                'Test_RMSE': results['test_metrics']['rmse'],
                'Test_MAPE': results['test_metrics']['mape'],
                'Performance_Drop': results['train_metrics']['r2'] - results['test_metrics']['r2']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create summary plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'R² Score Comparison',
                'RMSE Comparison', 
                'MAPE by Split',
                'Performance Drop (Train - Test R²)'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. R² comparison
        fig.add_trace(
            go.Bar(
                x=summary_df['Split'],
                y=summary_df['Train_R2'],
                name='Train R²',
                marker_color='blue'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=summary_df['Split'],
                y=summary_df['Test_R2'],
                name='Test R²',
                marker_color='red'
            ),
            row=1, col=1
        )
        
        # 2. RMSE comparison
        fig.add_trace(
            go.Bar(
                x=summary_df['Split'],
                y=summary_df['Train_RMSE'],
                name='Train RMSE',
                marker_color='blue',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=summary_df['Split'],
                y=summary_df['Test_RMSE'],
                name='Test RMSE',
                marker_color='red',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. MAPE
        fig.add_trace(
            go.Bar(
                x=summary_df['Split'],
                y=summary_df['Test_MAPE'],
                name='Test MAPE',
                marker_color='orange'
            ),
            row=2, col=1
        )
        
        # 4. Performance drop
        fig.add_trace(
            go.Bar(
                x=summary_df['Split'],
                y=summary_df['Performance_Drop'],
                name='Performance Drop',
                marker_color='purple'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Train-Test Splits Performance Summary',
            height=800,
            barmode='group'
        )
        
        # Save summary
        fig.write_html(os.path.join(self.output_dir, 'train_test_splits_summary.html'))
        summary_df.to_csv(os.path.join(self.output_dir, 'train_test_splits_summary.csv'), index=False)
        
        print(f"Summary saved to {self.output_dir}/train_test_splits_summary.html")
        print(f"Summary data saved to {self.output_dir}/train_test_splits_summary.csv")

def main():
    """Main function to run train-test splits analysis."""
    analyzer = TrainTestSplitsAnalyzer()
    analyzer.run_all_splits()

if __name__ == "__main__":
    main() 