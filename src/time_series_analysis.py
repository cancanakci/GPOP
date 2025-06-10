import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_time_series(start_date, years, freq='M'):
    """Generate a time series index for the specified number of years."""
    end_date = start_date + pd.DateOffset(years=years)
    return pd.date_range(start=start_date, end=end_date, freq=freq)

def create_feature_trend(historical_data, years, trend_type, trend_params):
    """
    Create a trend for a feature based on historical data and specified parameters.
    
    Parameters:
    - historical_data: Series containing historical values
    - years: Number of years to project
    - trend_type: Type of trend ('linear', 'exponential', 'polynomial')
    - trend_params: Dictionary containing trend parameters
        For linear: {'slope': float}
        For exponential: {'growth_rate': float}
        For polynomial: {'coefficients': list}
    """
    months = years * 12
    time_points = np.arange(months)
    
    # Get the last value from historical data as starting point
    initial_value = historical_data.iloc[-1]
    
    if trend_type == 'linear':
        slope = trend_params.get('slope', 0)
        return initial_value + slope * time_points
    
    elif trend_type == 'exponential':
        growth_rate = trend_params.get('growth_rate', 0)
        return initial_value * (1 + growth_rate) ** time_points
    
    elif trend_type == 'polynomial':
        coefficients = trend_params.get('coefficients', [0])
        return np.polyval(coefficients, time_points) + initial_value
    
    else:
        raise ValueError(f"Unsupported trend type: {trend_type}")

def create_scenario_dataframe(historical_data, years, feature_trends):
    """
    Create a dataframe with projected values for all features.
    
    Parameters:
    - historical_data: DataFrame containing historical values for all features
    - years: Number of years to project
    - feature_trends: Dictionary of trend configurations for each feature
    """
    # Get the last date from historical data
    start_date = historical_data.index[-1]
    dates = generate_time_series(start_date, years)
    
    # Create a new dataframe for future projections
    future_data = pd.DataFrame(index=dates)
    
    # For each feature, either apply the trend or keep the last value
    for feature in historical_data.columns:
        if feature in feature_trends:
            trend_config = feature_trends[feature]
            values = create_feature_trend(
                historical_data[feature],
                years,
                trend_config['type'],
                trend_config['params']
            )
            future_data[feature] = values
        else:
            # If no trend specified, keep the last value
            future_data[feature] = historical_data[feature].iloc[-1]
    
    # Combine historical and future data
    combined_data = pd.concat([historical_data, future_data])
    
    return combined_data

def plot_scenario(combined_data, years, title="Feature Trends Over Time"):
    """Create an interactive plot of the scenario data."""
    # Calculate the split point between historical and future data
    split_date = combined_data.index[-1] - pd.DateOffset(years=years)
    
    fig = make_subplots(rows=len(combined_data.columns), cols=1,
                       subplot_titles=combined_data.columns,
                       vertical_spacing=0.05)
    
    for i, column in enumerate(combined_data.columns, 1):
        # Plot historical data
        historical_mask = combined_data.index <= split_date
        fig.add_trace(
            go.Scatter(x=combined_data.index[historical_mask], 
                      y=combined_data[column][historical_mask],
                      name=f"{column} (Historical)",
                      mode='lines',
                      line=dict(color='blue')),
            row=i, col=1
        )
        
        # Plot future projections
        future_mask = combined_data.index > split_date
        fig.add_trace(
            go.Scatter(x=combined_data.index[future_mask], 
                      y=combined_data[column][future_mask],
                      name=f"{column} (Projected)",
                      mode='lines',
                      line=dict(color='red', dash='dash')),
            row=i, col=1
        )
        
        # Add vertical line to show split between historical and future
        fig.add_vline(x=split_date, line_dash="dash", line_color="gray", row=i, col=1)
        
        # Add hover information
        fig.update_traces(
            hovertemplate="Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
            row=i, col=1
        )
    
    fig.update_layout(
        height=300 * len(combined_data.columns),
        title_text=title,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def calculate_power_predictions(combined_data, model, scaler, feature_names):
    """Calculate power predictions for the scenario data."""
    # Scale the features
    scaled_features = scaler.transform(combined_data[feature_names])
    scaled_df = pd.DataFrame(scaled_features, columns=feature_names)
    
    # Make predictions
    predictions = model.predict(scaled_df)
    
    # Add predictions to the scenario data
    combined_data['Predicted Power Output (MW)'] = predictions
    
    return combined_data

def plot_power_predictions(combined_data, years):
    """Create an interactive plot of power predictions."""
    # Calculate the split point between historical and future data
    split_date = combined_data.index[-1] - pd.DateOffset(years=years)
    
    fig = go.Figure()
    
    # Plot historical predictions
    historical_mask = combined_data.index <= split_date
    fig.add_trace(
        go.Scatter(x=combined_data.index[historical_mask], 
                  y=combined_data['Predicted Power Output (MW)'][historical_mask],
                  name='Historical Predictions',
                  mode='lines',
                  line=dict(color='blue'))
    )
    
    # Plot future predictions
    future_mask = combined_data.index > split_date
    fig.add_trace(
        go.Scatter(x=combined_data.index[future_mask], 
                  y=combined_data['Predicted Power Output (MW)'][future_mask],
                  name='Future Projections',
                  mode='lines',
                  line=dict(color='red', dash='dash'))
    )
    
    # Add vertical line to show split between historical and future
    fig.add_vline(x=split_date, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title='Predicted Power Output Over Time',
        xaxis_title='Date',
        yaxis_title='Power Output (MW)',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig 