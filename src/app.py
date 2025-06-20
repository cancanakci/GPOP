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
from data_processing import load_and_parse, enforce_frequency, sanity_checks, prepare_nextday_input, create_nextday_features
from file_utils import load_latest_model_files, cleanup_old_models, load_default_model
from ui_components import display_data_visualizations, display_model_metrics, plot_scenario, display_time_series_analysis, clean_time_series_data, display_cleaning_summary
from core import load_selected_model_components, handle_prediction_workflow, create_scenario_dataframe, apply_well_drilling_strategy
from train import train_model, train_nextday_model

@st.cache_data
def load_default_data_cached():
    """Cache the default data loading to avoid redundant processing."""
    try:
        peek_df = pd.read_excel("data/default_data.xlsx", nrows=0)
        datetime_col = peek_df.columns[0]
        return load_and_parse("data/default_data.xlsx", datetime_col=datetime_col, silent=True)
    except Exception as e:
        st.error(f"Error loading default data: {e}")
        return None

@st.cache_data
def load_default_training_data_cached():
    """Cache the default training data loading."""
    try:
        return joblib.load(os.path.join("models", "default_training_data.pkl"))
    except Exception as e:
        st.error(f"Error loading default training data: {e}")
        return None

@st.cache_data
def load_default_model_components_cached():
    """Cache the default model components loading."""
    try:
        model = joblib.load(os.path.join("models", "default_model.pkl"))
        scaler = joblib.load(os.path.join("models", "default_scaler.pkl"))
        feature_names = joblib.load(os.path.join("models", "default_feature_names.pkl"))
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading default model components: {e}")
        return None, None, None

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="GPOP", page_icon="💨", layout="wide")
    st.title("Geothermal Power Output Prediction")

    # --- Main Tabs ---
    training_tab, analysis_tab, forecasting_tab, nextday_tab = st.tabs([
        "Model Training & Prediction", 
        "Time Series Analysis", 
        "Long Horizon Extrapolation",
        "Next Day Prediction"
    ])

    # --- Model Training & Prediction Tab ---
    with training_tab:
        st.header("Model Training & Prediction")
        
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # Load default model components using cached function
        model, scaler, feature_names, training_data, status = load_selected_model_components("Use Default Model", models_dir)
        st.sidebar.write(status)

        if model and scaler and feature_names and training_data:
            explore_tab, explore_clean_tab, predict_tab = st.tabs(["Explore Raw Data", "Explore Clean Data", "Make Predictions"])
            with explore_tab:
                display_data_visualizations(training_data, model)
            with explore_clean_tab:
                # For default model, clean the data on the fly
                metrics_path = os.path.join(models_dir, "default_metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    # Use cached data loading
                    default_data = load_default_data_cached()
                    if default_data is not None:
                        cleaned_default, cleaning_steps = clean_time_series_data(default_data)
                        target_col = metrics.get('target_column', cleaned_default.columns[-1])
                        display_data_visualizations({
                            'X_train': cleaned_default.drop(target_col, axis=1),
                            'y_train': cleaned_default[target_col],
                            'feature_names': cleaned_default.drop(target_col, axis=1).columns.tolist(),
                            'target_column': target_col
                        }, model)
                        display_cleaning_summary(cleaning_steps)
            with predict_tab:
                st.header("Make Predictions with the Loaded Model")
                handle_prediction_workflow(model, scaler, feature_names, training_data)

            # Display model metrics
            metrics_path = os.path.join(models_dir, "default_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                display_model_metrics(metrics)

    # --- Time Series Analysis Tab ---
    with analysis_tab:
        try:
            # Use cached data loading
            default_training_data = load_default_training_data_cached()
            model, scaler, feature_names = load_default_model_components_cached()
            default_data = load_default_data_cached()
            
            if default_training_data is not None and default_data is not None and model is not None:
                target_col = default_training_data.get('target_column', default_data.columns[-1])
                training_data_for_analysis = {
                    'X_train': default_data.drop(columns=[target_col]),
                    'y_train': default_data[target_col],
                    'target_column': target_col,
                    'scaler': scaler
                }

                raw_tab, clean_tab = st.tabs(["Raw Data", "Cleaned Data"])
                with raw_tab:
                    display_time_series_analysis(training_data_for_analysis, model, label_prefix="Raw ")
                with clean_tab:
                    # Clean the data for display
                    cleaned_X, cleaning_steps_X = clean_time_series_data(training_data_for_analysis['X_train'])
                    cleaned_y, cleaning_steps_y = clean_time_series_data(training_data_for_analysis['y_train'].to_frame())
                    cleaned_data = {
                        'X_train': cleaned_X,
                        'y_train': cleaned_y.squeeze(),
                        'target_column': training_data_for_analysis['target_column'],
                        'scaler': training_data_for_analysis.get('scaler')
                    }
                    display_time_series_analysis(cleaned_data, model, label_prefix="Cleaned ")
                    display_cleaning_summary(cleaning_steps_X + cleaning_steps_y)
        except Exception as e:
            st.error(f"An error occurred in the Time Series Analysis tab: {e}")
            st.exception(e)

    # --- Long Horizon Extrapolation Tab ---
    with forecasting_tab:
        st.header("Long Horizon Extrapolation")
        try:
            # Use cached data loading
            default_data = load_default_data_cached()
            default_training_data = load_default_training_data_cached()
            model, scaler, feature_names = load_default_model_components_cached()
            
            if default_data is not None and default_training_data is not None and model is not None and scaler is not None and feature_names is not None:
                ts_data = enforce_frequency(default_data, freq='h', silent=True)
                target_col = default_training_data.get('target_column', ts_data.columns[-1])
                brine_col = "Brine Flowrate (T/h)"
                steam_col = "NCG+Steam Flowrate (T/h)"

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
                    plot_scenario(st.session_state.scenario_data, st.session_state.years, st.session_state.target_col, st.session_state.feature_trends)
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
                        
                        st.subheader(f"Expected number of make-up wells to drill over {years} years: **{len(pulses)}**")

                        if 'csv_with_sim' in st.session_state:
                            st.download_button("Download Scenario Data (with MUW simulation)", st.session_state.csv_with_sim, "scenario_predictions_with_MUW_simulation.csv", "text/csv")
                
        except Exception as e:
            st.error(f"An error occurred in the forecasting tab: {e}")

    # --- Next Day Prediction Tab ---
    with nextday_tab:
        st.header("Next Day Prediction")
        st.write("Upload today's operational data (24 hours) to predict tomorrow's hourly gross power output.")
        
        # Show expected upload file format
        st.subheader("Expected Upload File Format")
        st.write("Your CSV/Excel file should contain exactly 24 rows (one for each hour) with the following columns:")
        
        # Create example data (updated to remove pressure differential)
        example_data = pd.DataFrame({
            'datetime': pd.date_range('2024-01-15 00:00:00', periods=24, freq='H'),
            'Brine Flowrate (T/h)': [1200 + i*2 for i in range(24)],
            'NCG+Steam Flowrate (T/h)': [450 - i*1.2 for i in range(24)],
            'Ambient Temperature (°C)': [15 - i*0.3 for i in range(24)],
        })
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("**Required columns:**")
            st.dataframe(example_data.head(6), use_container_width=True)
            st.write("*Note: The 'Gross Power (MW)' column is used for validation but not for prediction*")
        
        with col2:
            st.write("**File requirements:**")
            st.write("• Exactly 24 rows (hours)")
            st.write("• Hourly data (00:00 to 23:00)")
            st.write("• Datetime column (optional)")
            st.write("• All feature columns present")
        
        # Download example file
        csv_example = example_data.to_csv(index=False)
        st.download_button(
            "Download Example File (CSV)",
            data=csv_example,
            file_name="example_nextday_input.csv",
            mime="text/csv"
        )
        
        # Check if next-day model exists
        nextday_model_path = os.path.join(models_dir, "nextday_model.pkl")
        if not os.path.exists(nextday_model_path):
            st.warning("Next-day prediction model not found. Please train the model first.")
            if st.button("Train Next-Day Model"):
                with st.spinner("Training next-day prediction model..."):
                    try:
                        # Load default data for training
                        default_data = pd.read_excel("data/default_data.xlsx")
                        datetime_col = default_data.columns[0]
                        target_col = "Gross Power (MW)"  # Assuming this is the target
                        
                        # Train the next-day model
                        metrics = train_nextday_model(
                            data_source="data/default_data.xlsx",
                            models_dir=models_dir,
                            target_column=target_col,
                            datetime_col=datetime_col,
                            window_hours=24
                        )
                        
                        st.success("Next-day prediction model trained successfully!")
                        st.json(metrics)
                        
                    except Exception as e:
                        st.error(f"Error training next-day model: {e}")
                        st.exception(e)
        else:
            try:
                nextday_models = joblib.load(nextday_model_path)
                nextday_scaler = joblib.load(os.path.join(models_dir, "nextday_scaler.pkl"))
                nextday_feature_names = joblib.load(os.path.join(models_dir, "nextday_feature_names.pkl"))
                
                # Load metrics
                nextday_metrics_path = os.path.join(models_dir, "nextday_metrics.json")
                if os.path.exists(nextday_metrics_path):
                    with open(nextday_metrics_path, 'r') as f:
                        nextday_metrics = json.load(f)
                
                st.success("Next-day prediction model loaded successfully!")
                
                # Display model performance
                if 'nextday_metrics' in locals():
                    st.subheader("Model Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Overall RMSE", f"{nextday_metrics['overall_metrics']['rmse']:.4f}")
                    with col2:
                        st.metric("Overall R²", f"{nextday_metrics['overall_metrics']['r2']:.4f}")
                    with col3:
                        st.metric("Window Hours", nextday_metrics['window_hours'])
                    with col4:
                        st.metric("Best Test Day RMSE", f"{nextday_metrics.get('best_test_example_rmse', 'N/A'):.4f}")
                
                # --- Show best test example ---
                st.subheader("Best Prediction Example from Test Set")
                example_path = os.path.join(models_dir, "nextday_best_example.pkl")
                if os.path.exists(example_path):
                    try:
                        best_example = joblib.load(example_path)
                        
                        actual_values = best_example['actual_values']
                        predicted_values = best_example['predicted_values']
                        rmse = best_example['rmse']
                        input_date = pd.to_datetime(best_example['input_timestamp']).strftime('%Y-%m-%d')
                        prediction_date = (pd.to_datetime(best_example['input_timestamp']) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

                        st.write(f"This example shows the model's best performance on the test set, predicting for **{prediction_date}** based on data from **{input_date}**.")

                        # Create comparison DataFrame
                        hours = pd.date_range(start=prediction_date, periods=24, freq='H')
                        comparison_df = pd.DataFrame({
                            'Hour': hours,
                            'Actual (MW)': actual_values,
                            'Predicted (MW)': predicted_values,
                        })
                        comparison_df['Error (MW)'] = comparison_df['Predicted (MW)'] - comparison_df['Actual (MW)']
                        
                        # --- Visualization: Actual vs Predicted ---
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=comparison_df['Hour'],
                            y=comparison_df['Actual (MW)'],
                            mode='lines+markers',
                            name='Actual',
                            line=dict(color='#1f77b4', width=3),
                            marker=dict(size=8)
                        ))
                        fig.add_trace(go.Scatter(
                            x=comparison_df['Hour'],
                            y=comparison_df['Predicted (MW)'],
                            mode='lines+markers',
                            name='Predicted',
                            line=dict(color='#ff7f0e', width=3, dash='dash'),
                            marker=dict(size=8, symbol='diamond')
                        ))
                        
                        fig.update_layout(
                            title=f'Best Example: Actual vs Predicted Power ({prediction_date})',
                            xaxis_title='Hour',
                            yaxis_title='Gross Power (MW)',
                            yaxis_range=[0, 60], # Set fixed y-axis range
                            hovermode='x unified',
                            height=400,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # --- Visualization: Prediction Error Plot ---
                        fig_errors = go.Figure()
                        fig_errors.add_trace(go.Scatter(
                            x=list(range(24)),
                            y=comparison_df['Error (MW)'],
                            mode='lines+markers',
                            name=f'Error on {prediction_date}',
                            line=dict(color='#d62728', width=2),
                            marker=dict(size=6)
                        ))
                        fig_errors.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Perfect Prediction", annotation_position="bottom right")
                        fig_errors.update_layout(
                            title='Prediction Error by Hour of Day (Best Example)',
                            xaxis_title='Hour of Day',
                            yaxis_title='Prediction Error (MW)',
                            hovermode='x unified',
                            height=400
                        )
                        st.plotly_chart(fig_errors, use_container_width=True)

                        # --- Summary Statistics ---
                        st.write("**Prediction Summary for Best Example:**")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("RMSE", f"{rmse:.2f} MW")
                        col2.metric("Mean Absolute Error", f"{np.mean(np.abs(comparison_df['Error (MW)'].values)):.2f} MW")
                        col3.metric("Max Error", f"{comparison_df['Error (MW)'].abs().max():.2f} MW")
                        
                    except Exception as e:
                        st.warning(f"Could not load or display the best prediction example: {e}")
                else:
                    st.info("Best prediction example file not found. Please re-run the `create_default_model.py` script.")

                # File upload for today's data
                st.subheader("Make Your Prediction")
                uploaded_file = st.file_uploader(
                    "Upload today's operational data (CSV or Excel)", 
                    type=['csv', 'xlsx'], 
                    key="nextday_upload"
                )
                
                if uploaded_file:
                    try:
                        # Load and process the uploaded data
                        if uploaded_file.name.endswith('.csv'):
                            user_data = pd.read_csv(uploaded_file)
                        else:
                            user_data = pd.read_excel(uploaded_file)

                        # --- Data Validation ---
                        st.write("Uploaded Data Preview:")
                        st.dataframe(user_data.head(), use_container_width=True)

                        if len(user_data) != 24:
                            st.error(f"Error: Expected 24 rows (one for each hour), but got {len(user_data)}.")
                            st.stop()
                        
                        # Find datetime column if it exists
                        datetime_col = None
                        for col in user_data.columns:
                            if 'date' in col.lower() or 'time' in col.lower():
                                datetime_col = col
                                break
                        
                        if datetime_col:
                            try:
                                user_data[datetime_col] = pd.to_datetime(user_data[datetime_col])
                                user_data = user_data.set_index(datetime_col)
                            except Exception:
                                st.warning("Could not parse the datetime column. Proceeding without a time index.")
                                user_data = user_data.drop(columns=[datetime_col])
                        
                        # Check for required feature columns
                        base_features = [f.split('_hour_')[0] for f in nextday_feature_names if '_hour_' in f]
                        base_features = sorted(list(set(base_features))) # Get unique base features
                        
                        missing_cols = [col for col in base_features if col not in user_data.columns]
                        
                        if missing_cols:
                            st.error(f"Error: The following required columns are missing from your upload: {', '.join(missing_cols)}")
                            st.stop()

                        # --- Prepare input for prediction ---
                        # Add a dummy target column as it's expected by the processing function
                        user_data["Gross Power (MW)"] = 0
                        
                        # Select only the necessary columns in the correct order for the model
                        user_data_ordered = user_data[base_features + ["Gross Power (MW)"]]

                        input_features = create_nextday_features(user_data_ordered, "Gross Power (MW)", window_hours=24, silent=True)
                        
                        # Scale features
                        input_scaled = nextday_scaler.transform(input_features.drop(columns=['timestamp']))
                        input_scaled_df = pd.DataFrame(input_scaled, columns=nextday_feature_names)
                        
                        # Make predictions
                        predictions = []
                        for hour in range(24):
                            pred = nextday_models[hour].predict(input_scaled_df)[0]
                            predictions.append(pred)
                        
                        # --- Display Prediction Results ---
                        st.subheader("Your Next-Day Power Prediction")
                        
                        # Create results DataFrame
                        prediction_date = pd.to_datetime(user_data.index.max() if isinstance(user_data.index, pd.DatetimeIndex) else "today") + pd.Timedelta(days=1)
                        result_hours = pd.date_range(start=prediction_date.date(), periods=24, freq='H')
                        
                        results_df = pd.DataFrame({
                            'Hour': result_hours,
                            'Predicted Power (MW)': predictions
                        })
                        
                        # Plot results
                        fig_pred = go.Figure()
                        fig_pred.add_trace(go.Scatter(
                            x=results_df['Hour'],
                            y=results_df['Predicted Power (MW)'],
                            mode='lines+markers',
                            name='Predicted',
                            line=dict(color='#2ca02c', width=3),
                            marker=dict(size=8)
                        ))
                        
                        fig_pred.update_layout(
                            title=f'Predicted Gross Power for {prediction_date.strftime("%Y-%m-%d")}',
                            xaxis_title='Hour',
                            yaxis_title='Gross Power (MW)',
                            hovermode='x unified',
                            height=400
                        )
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.plotly_chart(fig_pred, use_container_width=True)
                        
                        with col2:
                            st.write("**Prediction Summary:**")
                            st.metric("Average Predicted Power", f"{np.mean(predictions):.2f} MW")
                            st.metric("Peak Power", f"{np.max(predictions):.2f} MW")
                            st.metric("Minimum Power", f"{np.min(predictions):.2f} MW")

                            st.write("**Predicted Hourly Values:**")
                            st.dataframe(results_df.set_index('Hour').round(2), use_container_width=True, height=250)
                            
                        # Download button for predictions
                        csv_pred = results_df.to_csv(index=False)
                        st.download_button(
                            "Download Predictions (CSV)",
                            data=csv_pred,
                            file_name=f"nextday_prediction_{prediction_date.strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )

                    except Exception as e:
                        st.error("An error occurred while processing your file.")
                        st.exception(e)
            except Exception as e:
                st.error(f"Error loading next-day model: {e}")
        
        # Add a button to retrain all models
        st.sidebar.subheader("Model Management")
        if st.sidebar.button("Re-train All Models", key="retrain_all"):
            with st.spinner("Re-training all models from `data/default_data.xlsx`... This may take a few minutes."):
                try:
                    import create_default_model
                    create_default_model.main()
                    st.success("All models have been re-trained successfully!")
                    st.rerun() # Rerun the app to load new models
                except Exception as e:
                    st.error(f"An error occurred during re-training: {e}")
                    st.exception(e)

def about_page():
    st.title("About GPOP")
    st.write("""
This application, **Geothermal Power Output Predictor (GPOP)**, is designed to provide forecasts for geothermal power generation.
It leverages machine learning models trained on historical operational data to deliver two key predictive functions:
- **Simple Power Prediction**: A straightforward model that predicts gross power output based on a given set of input features.
- **Next-Day Hourly Prediction**: A more complex model that takes 24 hours of operational data to forecast the hourly power output for the following 24 hours.

The models are trained using XGBoost, a powerful and efficient gradient boosting library. The application is built with Streamlit, providing an interactive and user-friendly interface.
    """)

if __name__ == "__main__":
    main()
