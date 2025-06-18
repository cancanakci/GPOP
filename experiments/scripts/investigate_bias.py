import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os

def analyze_bias():
    # Load data
    data = pd.read_excel("../../data/default_data.xlsx")
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    
    # 1. Analyze power output distribution by year
    print("=== Power Output Analysis by Year ===")
    yearly_stats = data.groupby(data['Datetime'].dt.year)['Gross Power (MW)'].agg(['mean', 'std', 'min', 'max'])
    print(yearly_stats)
    
    # 2. Train models and analyze error patterns
    print("\n=== Error Analysis by Training Year ===")
    train_years = [2020, 2021, 2022, 2023]
    error_analysis = {}
    
    for train_year in train_years:
        print(f"\nTraining on {train_year}:")
        
        # Split data
        train_data = data[data['Datetime'].dt.year == train_year]
        test_data = data[data['Datetime'].dt.year != train_year]
        
        # Prepare features
        features = ['Brine Flowrate (T/h)', 'Fluid Temperature (°C)', 'NCG+Steam Flowrate (T/h)', 
                   'Ambient Temperature (°C)', 'Reinjection Temperature (°C)']
        
        X_train = train_data[features]
        y_train = train_data['Gross Power (MW)']
        X_test = test_data[features]
        y_test = test_data['Gross Power (MW)']
        
        # Train model
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, 
                                min_child_weight=5, subsample=0.8, colsample_bytree=0.8, 
                                random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        train_preds = model.predict(X_train_scaled)
        test_preds = model.predict(X_test_scaled)
        
        # Calculate errors
        train_errors = y_train - train_preds
        test_errors = y_test - test_preds
        
        # Analyze errors by year
        test_data_with_errors = test_data.copy()
        test_data_with_errors['errors'] = test_errors
        
        error_by_year = test_data_with_errors.groupby(test_data_with_errors['Datetime'].dt.year)['errors'].agg(['mean', 'std'])
        
        print(f"  Training year {train_year} stats:")
        print(f"    Train data mean power: {y_train.mean():.2f}")
        print(f"    Train data mean error: {train_errors.mean():.2f}")
        print(f"    Test data mean error: {test_errors.mean():.2f}")
        print(f"    Error by test year:")
        for year, stats in error_by_year.iterrows():
            print(f"      {year}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        
        error_analysis[train_year] = {
            'train_mean_power': y_train.mean(),
            'train_mean_error': train_errors.mean(),
            'test_mean_error': test_errors.mean(),
            'error_by_year': error_by_year
        }
    
    # 3. Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Power Output by Year', 'Mean Error by Training Year', 
                       'Error Distribution by Year', 'Training vs Test Error'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Power output by year
    years = yearly_stats.index
    means = yearly_stats['mean']
    fig.add_trace(
        go.Bar(x=years, y=means, name='Mean Power Output'),
        row=1, col=1
    )
    
    # Plot 2: Mean error by training year
    train_years_list = list(error_analysis.keys())
    test_errors = [error_analysis[year]['test_mean_error'] for year in train_years_list]
    train_errors = [error_analysis[year]['train_mean_error'] for year in train_years_list]
    
    fig.add_trace(
        go.Bar(x=train_years_list, y=test_errors, name='Test Error'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=train_years_list, y=train_errors, name='Train Error'),
        row=1, col=2
    )
    
    # Plot 3: Error distribution by year for each training model
    colors = ['blue', 'red', 'green', 'orange']
    for i, train_year in enumerate(train_years_list):
        error_by_year = error_analysis[train_year]['error_by_year']
        fig.add_trace(
            go.Scatter(x=error_by_year.index, y=error_by_year['mean'], 
                      mode='lines+markers', name=f'Trained on {train_year}',
                      line=dict(color=colors[i])),
            row=2, col=1
        )
    
    # Plot 4: Training vs Test error
    fig.add_trace(
        go.Scatter(x=train_errors, y=test_errors, mode='markers+text',
                  text=train_years_list, textposition="top center",
                  name='Training vs Test Error'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Bias Analysis")
    fig.write_html("../plots/bias_analysis.html")
    print(f"\nBias analysis plot saved to ../plots/bias_analysis.html")
    
    return error_analysis, yearly_stats

if __name__ == "__main__":
    analyze_bias() 