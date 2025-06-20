"""
ui_components.py
----------------
Functions for creating and displaying Streamlit UI components and plots.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import colorsys
import numpy as np
import scipy.stats as stats

# Consistent color palette for features across the application (user's preferred colors + blue)
FEATURE_COLORS = [
    '#6a00fa',  # purple
    '#ff0e9a',  # pink
    '#ffa10a',  # orange
    '#ff436a',  # red
    '#009cfa',  # blue
]

def get_feature_color_map(features: list) -> dict:
    """Creates a consistent color mapping for a list of features. 'Gross Power (MW)' is always white."""
    color_map = {}
    color_idx = 0
    for feature in features:
        if feature == 'Gross Power (MW)':
            color_map[feature] = '#00ff00'
        else:
            color_map[feature] = FEATURE_COLORS[color_idx % len(FEATURE_COLORS)]
            color_idx += 1
    return color_map

def display_input_warnings(yellow_warnings, red_warnings, warning_flags_df=None, warning_ranges=None, input_df=None):
    """Displays input data warnings based on feature values being outside training data ranges."""
    is_single_prediction = warning_flags_df is not None and len(warning_flags_df) == 1

    if red_warnings:
        st.error("⚠️ **Input Validation Warning**: The following features have values outside the training data range. This may indicate data quality issues, unit mismatches, or invalid inputs. Please verify your data:")
        for feature in red_warnings:
            ranges = warning_ranges[feature]
            st.write(f"- **{feature}** (Training Range: {ranges['min']:.2f} - {ranges['max']:.2f})")

    if yellow_warnings:
        st.warning("⚠️ **Data Quality Note**: The following features have values outside the typical interquartile range. Please verify your data:")
        for feature in yellow_warnings:
            ranges = warning_ranges[feature]
            st.write(f"- **{feature}** (IQR Range: {ranges['iqr_lower']:.2f} - {ranges['iqr_upper']:.2f})")

    if warning_flags_df is not None and not is_single_prediction:
        total_rows = len(warning_flags_df)
        red_warning_rows = warning_flags_df['has_red_warning'].sum()
        yellow_warning_rows = warning_flags_df['has_yellow_warning'].sum()
        
        st.info(f"**Warning Summary**: {red_warning_rows} rows have validation warnings, {yellow_warning_rows} rows have quality warnings out of {total_rows} total rows.")

def display_data_visualizations(training_data, model=None):
    """Display various visualizations for the training and test data with comprehensive dataset information."""
    if training_data is None:
        return
    
    X_train = training_data.get('X_train')
    y_train = training_data.get('y_train')
    X_test = training_data.get('X_test')
    y_test = training_data.get('y_test')
    feature_names = training_data['feature_names']
    target_column = training_data.get('target_column', 'Target')
    
    st.header("Model & Training Data Exploration")
    
    # Dataset Overview Section
    st.subheader("Dataset Overview")
    
    # Create overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if X_train is not None:
            st.metric("Training Samples", len(X_train))
        else:
            st.metric("Training Samples", "N/A")
    
    with col2:
        if X_test is not None and len(X_test) > 0:
            st.metric("Test Samples", len(X_test))
        else:
            st.metric("Test Samples", "N/A")
    
    with col3:
        if X_train is not None:
            st.metric("Features", len(feature_names))
        else:
            st.metric("Features", "N/A")
    
    with col4:
        if X_train is not None:
            st.metric("Target Variable", target_column)
        else:
            st.metric("Target Variable", "N/A")
    
    # Time Series Information (if available)
    if X_train is not None and hasattr(X_train, 'index') and isinstance(X_train.index, pd.DatetimeIndex):
        st.subheader("Time Series Information")
        col1, col2 = st.columns(2)
        with col1:
            start_date = X_train.index.min()
            end_date = X_train.index.max()
            st.metric("Data Start Date", start_date.strftime('%Y-%m-%d %H:%M'))
            st.metric("Data End Date", end_date.strftime('%Y-%m-%d %H:%M'))
        with col2:
            time_span = end_date - start_date
            st.metric("Total Time Span", f"{time_span.days} days")
            st.metric("Data Frequency", str(X_train.index.freq) if X_train.index.freq else "Irregular")
    
    # Data Quality Information
    st.subheader("Data Quality Information")
    if X_train is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Training Set Statistics:**")
            if y_train is not None:
                train_stats = pd.DataFrame({
                    'Feature': [target_column] + feature_names,
                    'Min': [y_train.min()] + [X_train[col].min() for col in feature_names],
                    'Max': [y_train.max()] + [X_train[col].max() for col in feature_names],
                    'Mean': [y_train.mean()] + [X_train[col].mean() for col in feature_names],
                    'Std': [y_train.std()] + [X_train[col].std() for col in feature_names]
                })
                st.dataframe(train_stats, use_container_width=True)
        with col2:
            if X_test is not None and len(X_test) > 0:
                st.write("**Test Set Statistics:**")
                test_stats = pd.DataFrame({
                    'Feature': [target_column] + feature_names,
                    'Min': [y_test.min()] + [X_test[col].min() for col in feature_names],
                    'Max': [y_test.max()] + [X_test[col].max() for col in feature_names],
                    'Mean': [y_test.mean()] + [X_test[col].mean() for col in feature_names],
                    'Std': [y_test.std()] + [X_test[col].std() for col in feature_names]
                })
                st.dataframe(test_stats, use_container_width=True)
            else:
                st.info("No test set available for comparison.")
    # Feature Importance (if model is available)
    if model is not None and hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        fig = px.bar(feature_importance_df, 
                   x='Feature', 
                   y='Importance',
                   title='Feature Importance',
                   labels={'Importance': 'Relative Importance'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    # Correlation Analysis
    st.subheader("Feature Correlation Analysis")
    if X_train is not None:
        # Create combined dataframe for correlation analysis
        df_train = X_train.copy()
        if y_train is not None:
            df_train[target_column] = y_train.values
        corr = df_train.corr()
        fig = px.imshow(corr, text_auto=True, title='Feature Correlation Heatmap (Training Data)')
        fig.update_layout(
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            xaxis_nticks=len(corr.columns),
            yaxis_nticks=len(corr.index)
        )
        st.plotly_chart(fig, use_container_width=True)
    # Individual Feature Analysis
    st.subheader("Individual Feature Analysis")
    # Use training data for detailed analysis
    if X_train is not None and y_train is not None:
        df_analysis = X_train.copy()
        df_analysis[target_column] = y_train.values
        
        feature_color_map = get_feature_color_map(df_analysis.columns)

        for col in df_analysis.columns:
            fig = make_subplots(rows=1, cols=2, 
                              subplot_titles=(f'{col} Distribution', f'{col} Box Plot'))
            
            color = feature_color_map.get(col, '#1f77b4')
            fig.add_trace(go.Histogram(x=df_analysis[col], name='Histogram', marker_color=color), row=1, col=1)
            fig.add_trace(go.Box(y=df_analysis[col], name='Box Plot', marker_color=color), row=1, col=2)
            
            fig.update_layout(height=400, showlegend=False, title_text=f'{col} Analysis (Training Data)')
            st.plotly_chart(fig, use_container_width=True)

def display_prediction_visualizations(results_df, target_column='Target'):
    """Display visualizations for batch prediction results."""
    st.subheader("Prediction Results Visualizations")
    
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=('Prediction Distribution', 'Prediction Box Plot'))
    
    fig.add_trace(go.Histogram(x=results_df[f'Predicted {target_column}'], name='Histogram'), row=1, col=1)
    fig.add_trace(go.Box(y=results_df[f'Predicted {target_column}'], name='Box Plot'), row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False, title_text=f'Predicted {target_column} Analysis')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Time series plot if datetime column exists
    datetime_cols = [col for col in results_df.columns if any(term in col.lower() for term in ['date', 'time', 'tarih', 'datetime'])]
    if datetime_cols:
        datetime_col = datetime_cols[0]  # Use the first datetime column found
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(results_df[datetime_col]):
                results_df[datetime_col] = pd.to_datetime(results_df[datetime_col], errors='coerce')
            
            # Remove rows with invalid datetime
            valid_datetime_mask = results_df[datetime_col].notna()
            if valid_datetime_mask.sum() > 0:
                results_df_clean = results_df[valid_datetime_mask].copy()
                
                # Sort by datetime
                results_df_clean = results_df_clean.sort_values(datetime_col)
                
                # Create time series plot
                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(
                    x=results_df_clean[datetime_col],
                    y=results_df_clean[f'Predicted {target_column}'],
                    mode='lines+markers',
                    name='Predictions',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # Add warning indicators if any
                if 'Has Red Warning' in results_df_clean.columns:
                    red_warning_points = results_df_clean[results_df_clean['Has Red Warning']]
                    if not red_warning_points.empty:
                        fig_ts.add_trace(go.Scatter(
                            x=red_warning_points[datetime_col],
                            y=red_warning_points[f'Predicted {target_column}'],
                            mode='markers',
                            marker=dict(color='red', size=10, symbol='x'),
                            name='Data Quality Warnings'
                        ))
                
                fig_ts.update_layout(
                    title=f'Predicted {target_column} Over Time',
                    xaxis_title='Time',
                    yaxis_title=f'Predicted {target_column}',
                    hovermode='x unified'
                )
                st.plotly_chart(fig_ts, use_container_width=True)
            else:
                st.warning("No valid datetime values found in the datetime column.")
        except Exception as e:
            st.warning(f"Could not create time series plot: {str(e)}")
    
    # Feature correlations plot
    try:
        corr = results_df.corr()[f'Predicted {target_column}'].sort_values(ascending=False)
        fig = px.bar(x=corr.index, y=corr.values,
                    title='Feature Correlations with Predictions',
                    labels={'x': 'Features', 'y': 'Correlation'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not create correlation plot: {str(e)}")

def display_model_metrics(metrics):
    """Displays model metrics and visualizations in the sidebar (test set only)."""
    if not metrics:
        st.sidebar.warning("No model metrics found.")
        return

    st.sidebar.title("Model Information")
    st.sidebar.write(f"Model Type: {metrics.get('model_type', 'N/A')}")
    timestamp_str = metrics.get('timestamp')
    if timestamp_str:
        st.sidebar.write(f"Training Date: {datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')}")

    st.sidebar.subheader("Test Set Performance")

    # Only show test set metrics
    test_metrics = metrics.get('metrics', {})
    st.sidebar.write(f"R² Score: {test_metrics.get('r2', 0.0):.4f}")
    st.sidebar.write(f"RMSE: {test_metrics.get('rmse', 0.0):.4f}")
    st.sidebar.write(f"MSE: {test_metrics.get('mse', 0.0):.4f}")

    # Only plot actual vs predicted for the test set
    actual = metrics.get('actual')
    predicted = metrics.get('predicted')
    if actual and predicted:
        df_plot = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
        fig = px.scatter(df_plot, x='Actual', y='Predicted',
                         title='Test Set: Actual vs Predicted',
                         labels={'Actual': 'Actual Brüt Güç', 'Predicted': 'Predicted Brüt Güç'})
        min_val = min(min(actual), min(predicted))
        max_val = max(max(actual), max(predicted))
        fig.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color='black', dash='dash'))
        st.sidebar.plotly_chart(fig, use_container_width=True)

    loss_curve = metrics.get('metrics', {}).get('loss_curve', {})
    if loss_curve.get('train_loss') and loss_curve.get('test_loss'):
        epochs = range(1, len(loss_curve['train_loss']) + 1)
        df_loss = pd.DataFrame({'Epoch': epochs, 'Training Loss': loss_curve['train_loss'], 'Validation Loss': loss_curve['test_loss']})
        fig = px.line(df_loss, x='Epoch', y=['Training Loss', 'Validation Loss'],
                    title='Training and Validation Loss',
                    labels={'value': 'RMSE Loss', 'variable': 'Dataset'})
        st.sidebar.plotly_chart(fig, use_container_width=True)

def lighten_color(hex_color, amount=0.5):
    """Lighten the given hex color by the given amount (0=original, 1=white)."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    l = min(1, l + (1 - l) * amount)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

def plot_scenario(scenario_data, years, target_col=None, feature_trends=None):
    """Plots the scenario data with separate subplots for each feature."""
    if len(scenario_data) > 2000:
        plot_data = scenario_data.resample('D').mean()
    else:
        plot_data = scenario_data

    if feature_trends:
        st.markdown("**Applied Modifiers:**")
        for feature, trend in feature_trends.items():
            if trend['type'].lower() == 'linear':
                st.write(f"- {feature}: Linear ({trend['params']['slope']:.2f} units/year)")
            elif trend['type'].lower() == 'exponential':
                st.write(f"- {feature}: Exponential ({trend['params']['growth_rate']*100:.2f}%/year)")
            elif trend['type'].lower() == 'polynomial':
                st.write(f"- {feature}: Polynomial (coeffs: {trend['params']['coefficients']})")
            else:
                st.write(f"- {feature}: Constant")

    features = [col for col in plot_data.columns if col != target_col]
    n_features = len(features)
    row_heights = [0.5] + [0.5 / n_features] * n_features
    fig = make_subplots(
        rows=n_features + 1,
        cols=1,
        subplot_titles=[f"Predicted {target_col}"] + features,
        row_heights=row_heights,
        vertical_spacing=0.08
    )

    # Define a color palette for different features
    all_plot_features = [target_col] + features if target_col else features
    feature_color_map = get_feature_color_map(all_plot_features)
    
    # Calculate split date for historical vs projected data
    split_date = plot_data.index[-int(years * 365.25 * (pd.to_timedelta(plot_data.index.freq) / pd.Timedelta(days=1)))] if len(plot_data) > 0 and plot_data.index.freq else pd.Timestamp.now()

    # Plot target variable
    historical_target = plot_data[target_col][:split_date]
    future_target = plot_data[target_col][split_date:]
    base_color = feature_color_map.get(target_col, '#1f77b4')
    light_color = lighten_color(base_color, 0.5)
    fig.add_trace(go.Scatter(
        x=historical_target.index, 
        y=historical_target.values, 
        name=f'{target_col} (Historical)', 
        line=dict(color=base_color, width=3)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=future_target.index, 
        y=future_target.values, 
        name=f'{target_col} (Projected)', 
        line=dict(color=light_color, width=3, dash='dash')
    ), row=1, col=1)

    # Plot each feature with unique colors
    for i, feature in enumerate(features, 2):
        historical_feature = plot_data[feature][:split_date]
        future_feature = plot_data[feature][split_date:]
        
        # Use different color for each feature, cycling through the palette
        base_color = feature_color_map.get(feature, '#1f77b4')
        light_color = lighten_color(base_color, 0.5)
        
        # Historical data (solid line, base color)
        fig.add_trace(go.Scatter(
            x=historical_feature.index, 
            y=historical_feature.values, 
            name=f'{feature} (Historical)', 
            line=dict(color=base_color, width=2)
        ), row=i, col=1)
        
        # Projected data (dashed line, lighter color)
        fig.add_trace(go.Scatter(
            x=future_feature.index, 
            y=future_feature.values, 
            name=f'{feature} (Projected)', 
            line=dict(color=light_color, width=2, dash='dash')
        ), row=i, col=1)

    fig.update_layout(
        height=300 * (n_features + 1), 
        title_text=f"Scenario Projection ({years} years)",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

def display_time_series_analysis(training_data, model=None, label_prefix=""):
    """Display comprehensive time series analysis of the training data."""
    if training_data is None:
        st.warning("No training data available for time series analysis.")
        return
    # Check if we have time series data with datetime index
    if 'X_train' not in training_data or training_data['X_train'] is None:
        st.warning("Training data not available for time series analysis.")
        return
    X_train = training_data['X_train']
    y_train = training_data['y_train']
    target_column = training_data.get('target_column', 'Target')
    # Create a combined dataframe with target
    df_analysis = X_train.copy()
    df_analysis[target_column] = y_train.values
    # Check if we have a datetime index
    if not isinstance(df_analysis.index, pd.DatetimeIndex):
        st.warning("Data does not have a datetime index. Time series analysis requires time-indexed data.")
        return
    st.header("Time Series Analysis")
    # Basic time series information
    st.subheader("Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Observations", len(df_analysis))
    with col2:
        st.metric("Time Span", f"{(df_analysis.index.max() - df_analysis.index.min()).days} days")
    with col3:
        st.metric("Data Frequency", str(df_analysis.index.freq) if df_analysis.index.freq else "Irregular")
    # Create a consistent color map for all features
    feature_color_map = get_feature_color_map(df_analysis.columns)
    # Time series plots for all features
    st.subheader("Time Series Plots")
    # Target variable time series
    fig_target = go.Figure()
    fig_target.add_trace(go.Scatter(
        x=df_analysis.index,
        y=df_analysis[target_column],
        mode='lines',
        name=target_column,
        line=dict(color=feature_color_map.get(target_column), width=2)
    ))
    fig_target.update_layout(
        title=f'{label_prefix}{target_column} over time',
        xaxis_title='Time',
        yaxis_title=target_column,
        height=400
    )
    st.plotly_chart(fig_target, use_container_width=True)
    # Feature time series plots
    features = [col for col in df_analysis.columns if col != target_column]
    for feature in features:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_analysis.index,
            y=df_analysis[feature],
            mode='lines',
            name=feature,
            line=dict(width=1.5, color=feature_color_map.get(feature))
        ))
        fig.update_layout(
            title=f'{label_prefix}{feature} over time',
            xaxis_title='Time',
            yaxis_title=feature,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    # Trend Analysis
    st.subheader("Trend Analysis")
    # Linear trend analysis for each feature
    trend_results = {}
    for feature in df_analysis.columns:
        y = df_analysis[feature].values
        x = np.arange(len(y))
        # Fit linear trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        trend_results[feature] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'trend_direction': 'Increasing' if slope > 0 else 'Decreasing' if slope < 0 else 'No trend'
        }
    # Display trend results
    trend_df = pd.DataFrame(trend_results).T
    trend_df['slope'] = trend_df['slope'].astype(float).round(6)
    trend_df['r_squared'] = trend_df['r_squared'].astype(float).round(4)
    trend_df['p_value'] = trend_df['p_value'].astype(float).round(6)
    st.dataframe(trend_df[['slope', 'r_squared', 'p_value', 'trend_direction']])
    # Seasonality Analysis
    st.subheader("Seasonality Analysis")
    # Check for seasonality in target variable
    if len(df_analysis) > 24:  # Need sufficient data for seasonality analysis
        # Monthly seasonality
        monthly_avg = df_analysis[target_column].groupby(df_analysis.index.month).mean()
        fig_monthly = px.bar(
            x=monthly_avg.index,
            y=monthly_avg.values,
            title=f'Monthly Seasonality - {label_prefix}{target_column}',
            labels={'x': 'Month', 'y': f'Average {target_column}'}
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
        # Hourly seasonality (if hourly data)
        if df_analysis.index.freq and 'H' in str(df_analysis.index.freq):
            hourly_avg = df_analysis[target_column].groupby(df_analysis.index.hour).mean()
            fig_hourly = px.bar(
                x=hourly_avg.index,
                y=hourly_avg.values,
                title=f'Hourly Seasonality - {label_prefix}{target_column}',
                labels={'x': 'Hour', 'y': f'Average {target_column}'}
            )
            st.plotly_chart(fig_hourly, use_container_width=True)

def clean_time_series_data(
    df: pd.DataFrame,
    outlier_method: str = 'iqr',
    outlier_threshold: float = 1.5,
    rolling_window: int = 3
) -> (pd.DataFrame, list):
    """
    Cleans a time series DataFrame by:
    - Removing outliers (IQR or std method, replacing with NaN)
    - Filling missing values (forward fill, then backward fill)
    - Optionally smoothing with a rolling mean
    Returns (cleaned DataFrame, cleaning summary list)
    """
    df_clean = df.copy()
    cleaning_steps = []
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        if outlier_method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - outlier_threshold * IQR
            upper = Q3 + outlier_threshold * IQR
            outliers = (df_clean[col] < lower) | (df_clean[col] > upper)
            n_outliers = outliers.sum()
            df_clean.loc[outliers, col] = np.nan
            cleaning_steps.append(f"Removed {n_outliers} outliers in '{col}' using IQR method (threshold={outlier_threshold})")
        elif outlier_method == 'std':
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            lower = mean - outlier_threshold * std
            upper = mean + outlier_threshold * std
            outliers = (df_clean[col] < lower) | (df_clean[col] > upper)
            n_outliers = outliers.sum()
            df_clean.loc[outliers, col] = np.nan
            cleaning_steps.append(f"Removed {n_outliers} outliers in '{col}' using std method (threshold={outlier_threshold})")
        # Fill missing values
        n_missing = df_clean[col].isna().sum()
        df_clean[col] = df_clean[col].ffill().bfill()
        if n_missing > 0:
            cleaning_steps.append(f"Filled {n_missing} missing values in '{col}' with forward/backward fill")
        # Optional: smooth with rolling mean
        if rolling_window > 1:
            df_clean[col] = df_clean[col].rolling(window=rolling_window, min_periods=1, center=True).mean()
            cleaning_steps.append(f"Applied rolling mean smoothing to '{col}' (window={rolling_window})")
    return df_clean, cleaning_steps

def display_cleaning_summary(cleaning_steps: list):
    """Display a summary of cleaning steps in the UI."""
    if cleaning_steps:
        st.info("**Data Cleaning Summary:**\n" + "\n".join(f"- {step}" for step in cleaning_steps))

def display_nextday_prediction_examples(examples: list, title: str):
    """
    Displays the actual vs. predicted values for multiple next-day prediction examples on a single plot.
    
    Args:
        examples (list): A list of dictionaries, each containing prediction data.
        title (str): The title for the expander.
    """
    with st.expander(title, expanded=True):
        if not examples:
            st.warning("No prediction examples to display.")
            return

        # --- Chart 1: Actual vs. Predicted ---
        fig1 = go.Figure()
        
        # Define a color palette for the examples
        colors = px.colors.qualitative.Plotly

        for i, example_data in enumerate(examples):
            color = colors[i % len(colors)]
            prediction_date = pd.to_datetime(example_data['input_timestamp']) + pd.Timedelta(days=1)
            
            # Create a common set of hours for the x-axis
            hours = [f"{h:02d}:00" for h in range(24)]

            # Add predicted line
            fig1.add_trace(go.Scatter(
                x=hours, 
                y=example_data['predicted_values'], 
                name=f"Predicted (Example {i+1})",
                mode='lines+markers', 
                line=dict(color=color, dash='dash'),
                legendgroup=f"group{i}"
            ))
            
            # Add actual line
            fig1.add_trace(go.Scatter(
                x=hours, 
                y=example_data['actual_values'], 
                name=f"Actual (Example {i+1})",
                mode='lines', 
                line=dict(color=color, width=3),
                legendgroup=f"group{i}"
            ))

        fig1.update_layout(
            title="Prediction Examples from Test Set",
            xaxis_title="Hour of Day",
            yaxis_title="Gross Power (MW)",
            yaxis_range=[20, 60],  # Set fixed y-axis range
            legend_title="Predictions"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # --- Chart 2: Prediction Error ---
        fig2 = go.Figure()

        for i, example_data in enumerate(examples):
            color = colors[i % len(colors)]
            actuals = np.array(example_data['actual_values'])
            predicted = np.array(example_data['predicted_values'])
            error = predicted - actuals
            hours = [f"{h:02d}:00" for h in range(24)]

            fig2.add_trace(go.Bar(
                x=hours, 
                y=error, 
                name=f"Error (Example {i+1})",
                marker_color=color
            ))

        # Add a line for perfect prediction (zero error)
        fig2.add_hline(y=0, line_dash="dash", line_color="grey")

        fig2.update_layout(
            title="Prediction Error by Hour of Day",
            xaxis_title="Hour of Day",
            yaxis_title="Prediction Error (MW)",
            barmode='group'
        )
        st.plotly_chart(fig2, use_container_width=True)