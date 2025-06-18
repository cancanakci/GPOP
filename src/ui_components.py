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
    """Display various visualizations for the test data only."""
    if training_data is None:
        return
    
    X_test = training_data['X_test']
    y_test = training_data['y_test']
    feature_names = training_data['feature_names']
    target_column = training_data.get('target_column', 'Target')
    
    if X_test is None or len(X_test) == 0:
        st.info("No test set available. Visualizations will not be shown.")
        return
    
    df_test = X_test.copy()
    df_test[target_column] = y_test.values
    
    st.subheader("Test Set Data")
    
    if model is not None and hasattr(model, 'feature_importances_'):
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

    corr = df_test.corr()
    fig = px.imshow(corr, text_auto=True, title='Test Set Feature Correlation Heatmap')
    fig.update_layout(
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_nticks=len(corr.columns),
        yaxis_nticks=len(corr.index)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    for col in df_test.columns:
        fig = make_subplots(rows=1, cols=2, 
                          subplot_titles=(f'{col} Distribution', f'{col} Box Plot'))
        
        fig.add_trace(go.Histogram(x=df_test[col], name='Histogram'), row=1, col=1)
        fig.add_trace(go.Box(y=df_test[col], name='Box Plot'), row=1, col=2)
        
        fig.update_layout(height=400, showlegend=False, title_text=f'{col} Test Set Analysis')
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

    st.sidebar.subheader("Performance Metrics")

    # Always use test set metrics
    test_metrics = metrics.get('metrics', {})
    st.sidebar.write("Test Set Performance:")
    st.sidebar.write(f"R² Score: {test_metrics.get('r2', 0.0):.4f}")
    st.sidebar.write(f"RMSE: {test_metrics.get('rmse', 0.0):.4f}")
    st.sidebar.write(f"MSE: {test_metrics.get('mse', 0.0):.4f}")

    cv_metrics = metrics.get('metrics', {}).get('cv_metrics', {})
    if cv_metrics:
        st.sidebar.write("Cross-validation Performance:")
        st.sidebar.write(f"Mean R²: {cv_metrics.get('r2_mean', 0.0):.4f} (±{cv_metrics.get('r2_std', 0.0) * 2:.4f})")
        st.sidebar.write(f"Mean RMSE: {cv_metrics.get('rmse_mean', 0.0):.4f} (±{cv_metrics.get('rmse_std', 0.0) * 2:.4f})")

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
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Calculate split date for historical vs projected data
    split_date = plot_data.index[-int(years * 365.25 * (plot_data.index.freq / pd.Timedelta(days=1)))] if len(plot_data) > 0 else pd.Timestamp.now()

    # Plot target variable
    historical_target = plot_data[target_col][:split_date]
    future_target = plot_data[target_col][split_date:]
    base_color = colors[0]
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
        color_idx = (i - 1) % len(colors)
        base_color = colors[color_idx]
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