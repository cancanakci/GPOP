"""
Geothermal Power Output Prediction (GPOP)
-----------------------------------------
This Streamlit application provides a comprehensive interface for predicting
geothermal power output. It includes functionalities for model training,
batch and single predictions, and time series forecasting with well-drilling simulation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import json
import plotly.graph_objects as go

# Internal modules
from data_processing import load_and_parse, enforce_frequency, sanity_checks, detect_and_impute_outliers
from file_utils import load_latest_model_files, cleanup_old_models, load_default_model
from ui_components import display_data_visualizations, display_model_metrics, plot_scenario
from core import load_selected_model_components, handle_prediction_workflow, create_scenario_dataframe, apply_well_drilling_strategy
from train import train_model

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="GPOP", page_icon="💨", layout="wide")
    st.title("Geothermal Power Output Prediction")

    # --- Main Tabs ---
    training_tab, forecasting_tab = st.tabs(["Model Training & Prediction", "Time Series Forecasting"])

    # --- Model Training & Prediction Tab ---
    with training_tab:
        st.sidebar.title("Model Options")
        model_option = st.sidebar.radio("Choose an option:", ["Use Default Model", "Train New Model"])

        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        if model_option == "Train New Model":
            st.header("Train a New Model")
            training_file = st.file_uploader("Upload training data (CSV or Excel)", type=['csv', 'xlsx'], key="training_file")

            if training_file:
                try:
                    st.subheader("Time Series Settings")
                    
                    has_datetime_col = st.checkbox("My data has a datetime column", value=True)
                    
                    if has_datetime_col:
                        # Read the file header from the in-memory buffer to get columns
                        header_df = pd.read_excel(training_file, nrows=0) if training_file.name.lower().endswith(('.xlsx', '.xls')) else pd.read_csv(training_file, nrows=0)
                        # IMPORTANT: Reset buffer to the beginning for the next read
                        training_file.seek(0)
                        datetime_col = st.selectbox("Select your datetime column", header_df.columns.tolist())
                        start_date = None
                    else:
                        start_date = st.date_input("Select a start date for your data")
                        datetime_col = None

                    frequency = st.selectbox("Select data frequency", ["1min", "5min", "15min", "30min", "1h", "2h", "4h", "6h", "8h", "12h", "1d"], index=4)

                    df = load_and_parse(training_file, datetime_col=datetime_col, start_date=start_date, freq=frequency)
                    df = enforce_frequency(df, freq=frequency)
                    sanity_checks(df)
                    st.success("File loaded and preprocessed successfully!")
                    st.dataframe(df.head())

                    st.subheader("Column Mapping")
                    all_cols = df.columns.tolist()
                    target_col = st.selectbox("Select Target Column", all_cols, index=len(all_cols) - 1)
                    available_cols = [col for col in all_cols if col != target_col]
                    brine_col = st.selectbox("Select Primary Brine Flowrate Column", available_cols)
                    steam_col = st.selectbox("Select Primary Steam Flowrate Column", [c for c in available_cols if c != brine_col])

                    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                    if target_col in numeric_cols:
                        numeric_cols.remove(target_col)
                    df = detect_and_impute_outliers(df, cols=numeric_cols)

                    st.session_state['training_data_for_viz'] = {
                        'X_train': df.drop(target_col, axis=1), 'y_train': df[target_col],
                        'X_test': df.drop(target_col, axis=1).iloc[:0], 'y_test': df[target_col].iloc[:0], # For viz purposes
                        'feature_names': df.drop(target_col, axis=1).columns.tolist(),
                        'target_column': target_col, 'brine_col': brine_col, 'steam_col': steam_col
                    }

                    st.subheader("Model Hyperparameters")
                    col1, col2 = st.columns(2)
                    with col1:
                        n_estimators = st.slider("Number of Trees", 50, 800, 300, 50)
                        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.05, 0.01)
                        max_depth = st.slider("Max Tree Depth", 3, 10, 4, 1)
                    with col2:
                        min_child_weight = st.slider("Min Child Weight", 1, 10, 4, 1)
                        subsample = st.slider("Subsample Ratio", 0.5, 1.0, 0.8, 0.1)
                        colsample_bytree = st.slider("Column Sample by Tree", 0.5, 1.0, 0.8, 0.1)
                    
                    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.01)
                    st.session_state['model_params'] = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth, 'min_child_weight': min_child_weight, 'subsample': subsample, 'colsample_bytree': colsample_bytree}

                    if st.button("Train New Model"):
                        metrics = train_model(
                            training_file, 
                            models_dir, 
                            target_column=target_col, 
                            datetime_col=datetime_col,
                            start_date=start_date,
                            freq=frequency,
                            model_params=st.session_state.get('model_params'), 
                            test_size=test_size
                        )
                        st.session_state['new_model_metrics'] = metrics
                        st.session_state['new_model_trained'] = True
                        model, scaler, feature_names = load_latest_model_files(models_dir)
                        st.session_state['new_model_model'] = model
                        st.session_state['new_model_scaler'] = scaler
                        st.session_state['new_model_feature_names'] = feature_names
                        st.session_state['new_model_training_data'] = st.session_state['training_data_for_viz']
                        cleanup_old_models(models_dir)
                        st.success("New model trained and loaded successfully!")

                except Exception as e:
                    st.error(f"Error processing file: {e}")

        model, scaler, feature_names, training_data, status = load_selected_model_components(model_option, models_dir)
        st.sidebar.write(status)

        if model and scaler and feature_names and training_data:
            explore_tab, predict_tab = st.tabs(["Explore Model", "Make Predictions"])
            with explore_tab:
                st.header("Model & Training Data Exploration")
                display_data_visualizations(training_data, model)
            with predict_tab:
                st.header("Make Predictions with the Loaded Model")
                handle_prediction_workflow(model, scaler, feature_names, training_data)

            metrics = None
            if model_option == "Use Default Model":
                metrics_path = os.path.join(models_dir, "default_metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
            elif model_option == "Train New Model" and st.session_state.get('new_model_trained'):
                metrics = st.session_state.get('new_model_metrics')
            display_model_metrics(metrics)

    # --- Time Series Forecasting Tab ---
    with forecasting_tab:
        st.header("Time Series Forecasting")
        if not (model_option == "Use Default Model" or st.session_state.get('new_model_trained', False)):
            st.warning("Please train or select a model first to use Time Series Forecasting.")
        else:
            try:
                if model_option == "Use Default Model":
                    peek_df = pd.read_excel("data/default_data.xlsx", nrows=0)
                    datetime_col = peek_df.columns[0]
                    default_data = load_and_parse("data/default_data.xlsx", datetime_col=datetime_col)
                    ts_data = enforce_frequency(default_data, freq='h')
                    target_col = joblib.load(os.path.join("models", "default_training_data.pkl")).get('target_column', ts_data.columns[-1])
                    brine_col = "Brine Flowrate (T/h)"
                    steam_col = "NCG+Steam Flowrate (T/h)"
                else: # A new model is trained
                    ts_data = training_data['X_train'].copy()
                    ts_data[training_data['target_column']] = training_data['y_train']
                    target_col = training_data['target_column']
                    brine_col = training_data.get('brine_col')
                    steam_col = training_data.get('steam_col')

                st.subheader("Feature Trends")
                years = st.slider("Number of years to project", 1, 40, 20)
                default_trends = {
                    brine_col: {"type": "Exponential", "value": -3.0, "seasonality": True},
                    steam_col: {"type": "Exponential", "value": -3.0, "seasonality": True},
                    "Ambient Temperature (°C)": {"type": "Linear", "value": 1.0, "seasonality": True},
                    "Heat Exchanger Pressure Differential (Bar)": {"type": "Constant", "value": 0.0, "seasonality": False},
                }
                
                feature_trends = {}
                selected_features = [col for col in ts_data.columns if col != target_col]
                for feature in selected_features:
                    st.write(f"### {feature}")
                    defaults = default_trends.get(feature, {"type": "Constant", "value": 0.0, "seasonality": True})
                    trend_options = ["Constant", "Linear", "Exponential", "Polynomial"]
                    default_index = trend_options.index(defaults["type"]) if defaults["type"] in trend_options else 0
                    col1, col2 = st.columns(2)
                    with col1:
                        trend_type = st.selectbox("Select Trend Type", trend_options, index=default_index, key=f"trend_{feature}")
                    with col2:
                        add_seasonality = st.checkbox("Seasonality", value=defaults.get("seasonality", True), key=f"seasonality_{feature}")
                    feature_trends[feature] = {'type': trend_type, 'add_seasonality': add_seasonality}
                    
                    if trend_type == "Linear":
                        feature_trends[feature]['params'] = {'slope': st.number_input(f"Annual change for {feature}", value=defaults.get("value", 0.0), format="%.2f", key=f"slope_{feature}")}
                    elif trend_type == "Exponential":
                        feature_trends[feature]['params'] = {'growth_rate': st.number_input(f"Annual growth rate (%)", value=defaults.get("value", 0.0), format="%.2f", key=f"growth_{feature}") / 100}
                    elif trend_type == "Polynomial":
                        degree = st.slider(f"Polynomial degree", 1, 5, key=f"degree_{feature}")
                        coeffs = [st.number_input(f"Coefficient for x^{i}", 0.0, format="%.2f", key=f"coef_{feature}_{i}") for i in range(degree + 1)]
                        feature_trends[feature]['params'] = {'coefficients': coeffs}
                
                if st.button("Generate Scenario"):
                    scenario_features = create_scenario_dataframe(ts_data.drop(columns=[target_col]), years, feature_trends)
                    X_scaled = scaler.transform(scenario_features[feature_names])
                    predictions = model.predict(X_scaled)
                    scenario_data = scenario_features.copy()
                    scenario_data[target_col] = predictions
                    st.session_state.scenario_data = scenario_data.round(4)
                    st.session_state.years = years
                    st.session_state.target_col = target_col
                    st.session_state.brine_col = brine_col
                    st.session_state.steam_col = steam_col
                    st.session_state.feature_trends = feature_trends
                    st.session_state.csv_no_sim = scenario_data.to_csv()

                if 'scenario_data' in st.session_state:
                    st.subheader(f"Scenario Results")
                    st.plotly_chart(plot_scenario(st.session_state.scenario_data, st.session_state.years, st.session_state.target_col, st.session_state.feature_trends), use_container_width=True)
                    st.download_button("Download Scenario (no simulation)", st.session_state.csv_no_sim, "scenario_no_sim.csv", "text/csv")

                    st.subheader("Well Drilling Simulation")
                    threshold = st.number_input("Yearly average power threshold (MW)", 0, value=40, step=1)
                    muw_flowrate = st.number_input("Make-up Well Flowrate (T/h)", 0.0, value=400.0, step=10.0)
                    steam_percentage = st.number_input("Make-up Well Steam (%)", 0.0, 100.0, 10.0, 1.0)
                    
                    if st.button("Simulate New Wells"):
                        scenario_data = st.session_state.scenario_data
                        future_features = scenario_data[scenario_data.index > ts_data.index[-1]][selected_features]
                        future_power = scenario_data[scenario_data.index > ts_data.index[-1]][st.session_state.target_col]
                        
                        adjusted_power, pulses, adjusted_features = apply_well_drilling_strategy(future_features, future_power, model, scaler, feature_names, threshold, muw_flowrate, steam_percentage, st.session_state.brine_col, st.session_state.steam_col, st.session_state.feature_trends)
                        
                        adjusted_series = scenario_data[st.session_state.target_col].copy()
                        adjusted_series.loc[adjusted_series.index > ts_data.index[-1]] = adjusted_power
                        
                        st.session_state.adjusted_series = adjusted_series.round(4)
                        st.session_state.pulses = pulses
                        st.session_state.original_future_features = future_features.round(4)
                        st.session_state.adjusted_future_features = adjusted_features.round(4)

                        hist_features = scenario_data[scenario_data.index <= ts_data.index[-1]][selected_features]
                        full_adjusted_features = pd.concat([hist_features, adjusted_features])
                        download_df_sim = full_adjusted_features.copy()
                        download_df_sim[f'{target_col} (no MUW)'] = scenario_data[target_col]
                        download_df_sim[f'{target_col} (with MUW)'] = adjusted_series
                        st.session_state.csv_with_sim = download_df_sim.to_csv()

                if 'adjusted_series' in st.session_state:
                    st.subheader("Simulation Results")
                    adjusted_series = st.session_state.adjusted_series
                    pulses = st.session_state.pulses
                    original_future_features = st.session_state.original_future_features
                    adjusted_future_features = st.session_state.adjusted_future_features
                    scenario_data = st.session_state.scenario_data
                    target_col = st.session_state.target_col
                    years = st.session_state.years
                    brine_col = st.session_state.brine_col
                    steam_col = st.session_state.steam_col
                    
                    fig_well = go.Figure()
                    if len(scenario_data) > 2000:
                        plot_target = scenario_data[target_col].resample('D').mean()
                        plot_adjusted = adjusted_series.resample('D').mean()
                    else:
                        plot_target = scenario_data[target_col]
                        plot_adjusted = adjusted_series
                    fig_well.add_trace(go.Scatter(x=plot_adjusted.index, y=plot_adjusted.values, name='Adjusted with New Wells', mode='lines'))
                    fig_well.add_trace(go.Scatter(x=plot_target.index, y=plot_target.values, name='Original Predictions', mode='lines'))
                    for pulse_time in pulses:
                        fig_well.add_vline(x=pulse_time, line_color="red")
                    fig_well.update_layout(title='Power Predictions with New Wells', xaxis_title='Date', yaxis_title=target_col, hovermode='x unified')
                    st.plotly_chart(fig_well, use_container_width=True)

                    st.subheader("Yearly Power Predictions with New Wells")
                    yearly_adjusted_avg = adjusted_series.resample('Y').mean()
                    fig_yearly_well = go.Figure()
                    fig_yearly_well.add_trace(go.Scatter(x=yearly_adjusted_avg.index, y=yearly_adjusted_avg.round(4).values, name='Yearly Average', mode='lines'))
                    for pulse_time in pulses:
                        fig_yearly_well.add_vline(x=pulse_time, line_color="red")
                    fig_yearly_well.add_hline(y=threshold, line_dash="dot", line_color="red", annotation_text=f"Drilling Threshold: {threshold:.2f} MW", annotation_position="bottom right")
                    fig_yearly_well.update_layout(title='Yearly Average Power Predictions with New Wells', xaxis_title='Year', yaxis_title=f'Average {target_col}', hovermode='x unified')
                    st.plotly_chart(fig_yearly_well, use_container_width=True)

                    st.subheader("Quarterly Power Predictions with New Wells")
                    quarterly_adjusted_avg = adjusted_series.resample('3M').mean()
                    fig_quarterly_well = go.Figure()
                    fig_quarterly_well.add_trace(go.Scatter(x=quarterly_adjusted_avg.index, y=quarterly_adjusted_avg.round(4).values, name='Quarterly Average', mode='lines'))
                    for pulse_time in pulses:
                        fig_quarterly_well.add_vline(x=pulse_time, line_color="red")
                    fig_quarterly_well.add_hline(y=threshold, line_dash="dot", line_color="red", annotation_text=f"Drilling Threshold: {threshold:.2f} MW", annotation_position="bottom right")
                    fig_quarterly_well.update_layout(title='Quarterly Average Power Predictions with New Wells', xaxis_title='Quarter', yaxis_title=f'Average {target_col}', hovermode='x unified')
                    st.plotly_chart(fig_quarterly_well, use_container_width=True)

                    st.subheader("Adjusted Input Features from New Wells")
                    if len(original_future_features) > 2000:
                        plot_future_features = original_future_features.resample('D').mean()
                        plot_adjusted_features = adjusted_future_features.resample('D').mean()
                    else:
                        plot_future_features = original_future_features
                        plot_adjusted_features = adjusted_future_features
                    
                    fig_brine = go.Figure()
                    fig_brine.add_trace(go.Scatter(x=plot_adjusted_features.index, y=plot_adjusted_features[brine_col], name=f'Adjusted {brine_col}', mode='lines', line=dict(color='orange')))
                    fig_brine.add_trace(go.Scatter(x=plot_future_features.index, y=plot_future_features[brine_col], name=f'Original {brine_col}', mode='lines', line=dict(color='blue')))
                    for pulse_time in pulses:
                        fig_brine.add_vline(x=pulse_time, line_color="red")
                    fig_brine.update_layout(title=f'{brine_col} with New Wells', xaxis_title='Date', yaxis_title='Flowrate (T/h)', hovermode='x unified')
                    st.plotly_chart(fig_brine, use_container_width=True)

                    fig_ncg = go.Figure()
                    fig_ncg.add_trace(go.Scatter(x=plot_adjusted_features.index, y=plot_adjusted_features[steam_col], name=f'Adjusted {steam_col}', mode='lines', line=dict(color='purple')))
                    fig_ncg.add_trace(go.Scatter(x=plot_future_features.index, y=plot_future_features[steam_col], name=f'Original {steam_col}', mode='lines', line=dict(color='green')))
                    for pulse_time in pulses:
                        fig_ncg.add_vline(x=pulse_time, line_color="red")
                    fig_ncg.update_layout(title=f'{steam_col} with New Wells', xaxis_title='Date', yaxis_title='Flowrate (T/h)', hovermode='x unified')
                    st.plotly_chart(fig_ncg, use_container_width=True)
                    
                    st.subheader(f"Number of make-up wells to drill over {years} years: **{len(pulses)}**")

                    if 'csv_with_sim' in st.session_state:
                        st.download_button("Download Scenario Data (with MUW simulation)", st.session_state.csv_with_sim, "scenario_predictions_with_MUW_simulation.csv", "text/csv")
                    
            except Exception as e:
                st.error(f"An error occurred in the forecasting tab: {e}")

if __name__ == "__main__":
    main()
