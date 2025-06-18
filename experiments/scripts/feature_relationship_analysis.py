import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os

def analyze_feature_relationships():
    # Load data
    data = pd.read_excel("../../data/default_data.xlsx")
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data['Year'] = data['Datetime'].dt.year
    
    # Handle missing values
    data = data.ffill().bfill()
    
    # Check for any remaining NaNs
    print("Checking for NaN values:")
    for col in data.columns:
        nan_count = data[col].isna().sum()
        if nan_count > 0:
            print(f"  {col}: {nan_count} NaN values")
    
    features = ['Brine Flowrate (T/h)', 'Fluid Temperature (°C)', 'NCG+Steam Flowrate (T/h)', 
               'Ambient Temperature (°C)', 'Reinjection Temperature (°C)']
    target = 'Gross Power (MW)'
    
    print("=== Feature-Target Relationship Analysis Over Time ===\n")
    
    # 1. Correlation analysis by year
    print("1. CORRELATION ANALYSIS BY YEAR:")
    print("=" * 50)
    
    yearly_correlations = {}
    for year in [2020, 2021, 2022, 2023]:
        year_data = data[data['Year'] == year]
        correlations = year_data[features + [target]].corr()[target].drop(target)
        yearly_correlations[year] = correlations
        print(f"\nYear {year} (n={len(year_data)}):")
        for feature, corr in correlations.items():
            print(f"  {feature}: {corr:.4f}")
    
    # 2. Linear regression coefficients by year
    print("\n\n2. LINEAR REGRESSION COEFFICIENTS BY YEAR:")
    print("=" * 50)
    
    yearly_coefficients = {}
    for year in [2020, 2021, 2022, 2023]:
        year_data = data[data['Year'] == year]
        if len(year_data) == 0:
            continue
            
        X = year_data[features]
        y = year_data[target]
        
        # Check for NaN values in this year's data
        if X.isna().any().any() or y.isna().any():
            print(f"Warning: NaN values found in year {year}")
            print(f"  X NaN count: {X.isna().sum().sum()}")
            print(f"  y NaN count: {y.isna().sum()}")
            # Remove rows with NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            print(f"  After cleaning: {len(X)} samples")
        
        if len(X) == 0:
            print(f"No valid data for year {year}")
            continue
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit linear regression
        lr = LinearRegression()
        lr.fit(X_scaled, y)
        
        yearly_coefficients[year] = {
            'coefficients': lr.coef_,
            'intercept': lr.intercept_,
            'r2': lr.score(X_scaled, y),
            'mean_power': y.mean(),
            'std_power': y.std()
        }
        
        print(f"\nYear {year}:")
        print(f"  R²: {lr.score(X_scaled, y):.4f}")
        print(f"  Intercept: {lr.intercept_:.4f}")
        print(f"  Mean Power: {y.mean():.2f} MW")
        print(f"  Std Power: {y.std():.2f} MW")
        print("  Coefficients (standardized):")
        for feature, coef in zip(features, lr.coef_):
            print(f"    {feature}: {coef:.4f}")
    
    # 3. Feature distributions by year
    print("\n\n3. FEATURE DISTRIBUTIONS BY YEAR:")
    print("=" * 50)
    
    for feature in features:
        print(f"\n{feature}:")
        for year in [2020, 2021, 2022, 2023]:
            year_data = data[data['Year'] == year]
            if len(year_data) > 0:
                values = year_data[feature]
                print(f"  {year}: mean={values.mean():.2f}, std={values.std():.2f}, min={values.min():.2f}, max={values.max():.2f}")
    
    # 4. Create visualizations
    create_relationship_plots(data, features, target, yearly_correlations, yearly_coefficients)
    
    return yearly_correlations, yearly_coefficients

def create_relationship_plots(data, features, target, yearly_correlations, yearly_coefficients):
    """Create comprehensive plots showing feature-target relationships over time."""
    
    # 1. Correlation heatmap over time
    fig_corr = go.Figure()
    
    years = [2020, 2021, 2022, 2023]
    for year in years:
        if year in yearly_correlations:
            correlations = yearly_correlations[year]
            fig_corr.add_trace(go.Bar(
                x=features,
                y=correlations.values,
                name=f'Year {year}',
                text=[f'{v:.3f}' for v in correlations.values],
                textposition='auto'
            ))
    
    fig_corr.update_layout(
        title='Feature-Target Correlations Over Time',
        xaxis_title='Features',
        yaxis_title='Correlation with Power Output',
        barmode='group',
        height=500
    )
    fig_corr.write_html("../plots/feature_correlations_over_time.html")
    
    # 2. Regression coefficients over time
    fig_coeff = go.Figure()
    
    for i, feature in enumerate(features):
        coeffs = []
        years_available = []
        for year in years:
            if year in yearly_coefficients:
                coeffs.append(yearly_coefficients[year]['coefficients'][i])
                years_available.append(year)
        
        fig_coeff.add_trace(go.Scatter(
            x=years_available,
            y=coeffs,
            mode='lines+markers',
            name=feature,
            text=[f'{v:.3f}' for v in coeffs],
            textposition='top center'
        ))
    
    fig_coeff.update_layout(
        title='Standardized Regression Coefficients Over Time',
        xaxis_title='Year',
        yaxis_title='Coefficient Value',
        height=500
    )
    fig_coeff.write_html("../plots/regression_coefficients_over_time.html")
    
    # 3. Scatter plots for each feature vs target by year
    fig_scatter = make_subplots(
        rows=2, cols=3,
        subplot_titles=features,
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = ['blue', 'red', 'green', 'orange']
    for i, feature in enumerate(features):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        for j, year in enumerate([2020, 2021, 2022, 2023]):
            year_data = data[data['Year'] == year]
            if len(year_data) > 0:
                fig_scatter.add_trace(
                    go.Scatter(
                        x=year_data[feature],
                        y=year_data[target],
                        mode='markers',
                        name=f'Year {year}',
                        marker=dict(color=colors[j], opacity=0.6, size=3),
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=row, col=col
                )
    
    fig_scatter.update_layout(height=600, title_text="Feature vs Power Output by Year")
    fig_scatter.write_html("../plots/feature_vs_power_scatter.html")
    
    # 4. Power output distribution by year
    fig_dist = go.Figure()
    
    for year in [2020, 2021, 2022, 2023]:
        year_data = data[data['Year'] == year]
        if len(year_data) > 0:
            fig_dist.add_trace(go.Histogram(
                x=year_data[target],
                name=f'Year {year}',
                opacity=0.7,
                nbinsx=50
            ))
    
    fig_dist.update_layout(
        title='Power Output Distribution by Year',
        xaxis_title='Gross Power (MW)',
        yaxis_title='Frequency',
        barmode='overlay',
        height=500
    )
    fig_dist.write_html("../plots/power_distribution_by_year.html")
    
    # 5. Summary statistics table
    summary_data = []
    for year in [2020, 2021, 2022, 2023]:
        if year in yearly_coefficients:
            coeffs = yearly_coefficients[year]
            summary_data.append({
                'Year': year,
                'R²': coeffs['r2'],
                'Mean_Power': coeffs['mean_power'],
                'Std_Power': coeffs['std_power'],
                'Brine_Flowrate_Coef': coeffs['coefficients'][0],
                'Fluid_Temp_Coef': coeffs['coefficients'][1],
                'NCG_Steam_Coef': coeffs['coefficients'][2],
                'Ambient_Temp_Coef': coeffs['coefficients'][3],
                'Reinjection_Temp_Coef': coeffs['coefficients'][4]
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("../plots/feature_relationship_summary.csv", index=False)
    
    print(f"\nPlots saved to ../plots/")
    print(f"Summary table saved to ../plots/feature_relationship_summary.csv")

if __name__ == "__main__":
    analyze_feature_relationships() 