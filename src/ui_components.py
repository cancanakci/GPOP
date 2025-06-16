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
import numpy as np

def display_input_warnings(yellow_warnings, red_warnings, warning_flags_df=None, warning_ranges=None, input_df=None):
    """Displays input data warnings based on feature values being outside training data ranges."""
    is_single_prediction = warning_flags_df is not None and len(warning_flags_df) == 1

    if red_warnings:
        st.error("⚠️ Warning: The following features have values outside the training data range. This is usually caused by mismatched units, NaN inputs or mismatched features. Please verify values:")
        for feature in red_warnings:
            ranges = warning_ranges[feature]
            st.write(f"- {feature} (MIN/MAX Range: {ranges['min']:.2f} - {ranges['max']:.2f})")

    if yellow_warnings:
        st.warning("⚠️ Note: The following features have values outside the typical interquartile range. Please verify values:")
        for feature in yellow_warnings:
            ranges = warning_ranges[feature]
            st.write(f"- {feature} (IQR Range: {ranges['iqr_lower']:.2f} - {ranges['iqr_upper']:.2f})")

    if warning_flags_df is not None and not is_single_prediction:
        total_rows = len(warning_flags_df)
        red_warning_rows = warning_flags_df['has_red_warning'].sum()
        yellow_warning_rows = warning_flags_df['has_yellow_warning'].sum()
        
        st.info(f"Warning Summary: {red_warning_rows} rows have red warnings, {yellow_warning_rows} rows have yellow warnings out of {total_rows} total rows.")

def display_data_visualizations(training_data, model):
    """
    Displays various visualizations for model exploration, including feature importance
    and time series plots of the training data.
    """
    st.subheader("Feature Importance")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = training_data.get('feature_names', [])
        if feature_names:
            importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            importance_df = importance_df.sort_values(by='importance', ascending=False)
            
            fig = px.bar(importance_df, x='importance', y='feature', orientation='h', title='Model Feature Importance')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature names not available in training data.")
    else:
        st.info("This model type does not expose feature importances.")

    st.subheader("Training Data Time Series")
    try:
        # Reconstruct the full dataset from train/test splits to plot
        X_train = training_data['X_train']
        y_train = training_data['y_train']
        X_test = training_data['X_test']
        y_test = training_data['y_test']

        # Combine, sort, and assign markers for plotting
        y_full = pd.concat([y_train, y_test]).sort_index()
        X_full = pd.concat([X_train, X_test]).sort_index()
        
        # Plot the target variable
        st.write(f"**Target Variable: {training_data['target_column']}**")
        fig_target = go.Figure()
        fig_target.add_trace(go.Scatter(x=y_train.index, y=y_train.values, name='Train', mode='lines', line=dict(color='blue')))
        fig_target.add_trace(go.Scatter(x=y_test.index, y=y_test.values, name='Test', mode='lines', line=dict(color='orange')))
        fig_target.update_layout(title=f"Target Variable Over Time", xaxis_title='Date', yaxis_title=training_data['target_column'], hovermode='x unified')
        st.plotly_chart(fig_target, use_container_width=True)

        # Plot a selected feature
        st.write("**Feature Variables**")
        feature_to_plot = st.selectbox("Select a feature to visualize:", X_full.columns.tolist())
        
        if feature_to_plot:
            fig_feature = go.Figure()
            fig_feature.add_trace(go.Scatter(x=X_train.index, y=X_train[feature_to_plot], name='Train', mode='lines', line=dict(color='blue')))
            fig_feature.add_trace(go.Scatter(x=X_test.index, y=X_test[feature_to_plot], name='Test', mode='lines', line=dict(color='orange')))
            fig_feature.update_layout(title=f"'{feature_to_plot}' Over Time", xaxis_title='Date', yaxis_title=feature_to_plot, hovermode='x unified')
            st.plotly_chart(fig_feature, use_container_width=True)

    except KeyError as e:
        st.error(f"Could not generate time series plots. The training data is missing a required key: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while creating time series plots: {e}")

def display_prediction_visualizations(results_df, target_column='Target'):
    """Display visualizations for batch prediction results."""
    st.subheader("Prediction Results Visualizations")
    
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=('Prediction Distribution', 'Prediction Box Plot'))
    
    fig.add_trace(go.Histogram(x=results_df[f'Predicted {target_column}'], name='Histogram'), row=1, col=1)
    fig.add_trace(go.Box(y=results_df[f'Predicted {target_column}'], name='Box Plot'), row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False, title_text=f'Predicted {target_column} Analysis')
    
    st.plotly_chart(fig, use_container_width=True)
    
    if 'Tarih' in results_df.columns:
        fig = px.line(results_df, x='Tarih', y=f'Predicted {target_column}',
                     title=f'Predicted {target_column} Over Time')
        st.plotly_chart(fig, use_container_width=True)
    
    corr = results_df.corr()[f'Predicted {target_column}'].sort_values(ascending=False)
    fig = px.bar(x=corr.index, y=corr.values,
                title='Feature Correlations with Predictions',
                labels={'x': 'Features', 'y': 'Correlation'})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def display_model_metrics(metrics):
    """Displays model metrics and visualizations in the sidebar."""
    if not metrics:
        st.sidebar.warning("No model metrics found.")
        return

    st.sidebar.title("Model Information")
    st.sidebar.write(f"Model Type: {metrics.get('model_type', 'N/A')}")
    timestamp_str = metrics.get('timestamp')
    if timestamp_str:
        st.sidebar.write(f"Training Date: {datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')}")

    st.sidebar.subheader("Performance Metrics")

    test_metrics = metrics.get('metrics', {}).get('test_metrics', {}) or metrics.get('metrics', {})
    st.sidebar.write("Test Set Performance:")
    st.sidebar.write(f"R² Score: {test_metrics.get('r2', 0.0):.4f}")
    st.sidebar.write(f"RMSE: {test_metrics.get('rmse', 0.0):.4f}")
    st.sidebar.write(f"MSE: {test_metrics.get('mse', 0.0):.4f}")

    cv_metrics = metrics.get('metrics', {}).get('cv_metrics', {})
    if cv_metrics:
        st.sidebar.write("Cross-validation Performance:")
        st.sidebar.write(f"Mean R²: {cv_metrics.get('r2_mean', 0.0):.4f} (±{cv_metrics.get('r2_std', 0.0) * 2:.4f})")
        st.sidebar.write(f"Mean RMSE: {cv_metrics.get('rmse_mean', 0.0):.4f} (±{cv_metrics.get('rmse_std', 0.0) * 2:.4f})")

    actual = metrics.get('actual')
    predicted = metrics.get('predicted')
    if actual and predicted:
        df_plot = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
        fig = px.scatter(df_plot, x='Actual', y='Predicted',
                         title='Actual vs Predicted',
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

    split_date = plot_data.index[-int(years * 365.25 * (plot_data.index.freq / pd.Timedelta(days=1)))] if len(plot_data) > 0 else pd.Timestamp.now()

    historical_target = plot_data[target_col][:split_date]
    future_target = plot_data[target_col][split_date:]
    fig.add_trace(go.Scatter(x=historical_target.index, y=historical_target.values, name=f'{target_col} (Historical)', line=dict(color='blue', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=future_target.index, y=future_target.values, name=f'{target_col} (Predicted)', line=dict(color='lightblue', dash='dash', width=3)), row=1, col=1)

    for i, feature in enumerate(features, 2):
        historical_data = plot_data[feature][:split_date]
        future_data = plot_data[feature][split_date:]
        fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data.values, name=f'{feature} (Historical)'), row=i, col=1)
        fig.add_trace(go.Scatter(x=future_data.index, y=future_data.values, name=f'{feature} (Projected)', line=dict(dash='dash')), row=i, col=1)

    fig.update_layout(
        title=f'Feature Trends and {target_col} Prediction',
        height=400 + 200 * n_features,
        showlegend=True,
        hovermode='x unified'
    )
    fig.update_yaxes(title_text=target_col, row=1, col=1)
    for i, feature in enumerate(features, 2):
        fig.update_yaxes(title_text=feature, row=i, col=1)

    return fig 