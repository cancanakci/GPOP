"""
core.py
-------
Core application logic and workflows, separated from the UI.
"""
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import joblib
import os
import time
import plotly.graph_objects as go

from predict import predict, check_input_values
from file_utils import load_default_model
from ui_components import display_input_warnings, display_prediction_visualizations

def load_selected_model_components(model_option, models_dir):
    """Loads the default model components."""
    model, scaler, feature_names, training_data, status = None, None, None, None, ""

    try:
        model, scaler, feature_names = load_default_model(models_dir)
        if model and scaler and feature_names:
            try:
                training_data = joblib.load(os.path.join(models_dir, "default_training_data.pkl"))
                status = "Default model loaded successfully."
            except Exception as e:
                status = f"Default model loaded, but error loading training data: {e}"
        else:
            status = "Default model not found. Please ensure the model files exist."
    except Exception as e:
        status = f"Error loading default model: {e}"

    return model, scaler, feature_names, training_data, status

def handle_prediction_workflow(model, scaler, feature_names, training_data):
    """Handles the prediction workflow for both single and batch predictions."""
    target_column = training_data.get('target_column', 'Target')
    
    st.write("### Single Prediction")
    st.write("Enter values for each feature to get a prediction:")
    
    # Create input fields for each feature
    input_data = {}
    col1, col2 = st.columns(2)
    
    for i, feature in enumerate(feature_names):
        with col1 if i % 2 == 0 else col2:
            # Get min/max values from training data for better input validation
            min_val = training_data['X_train'][feature].min()
            max_val = training_data['X_train'][feature].max()
            mean_val = training_data['X_train'][feature].mean()
            
            input_data[feature] = st.number_input(
                f"{feature}",
                min_value=float(min_val * 0.5),
                max_value=float(max_val * 1.5),
                value=float(mean_val),
                step=float((max_val - min_val) / 100),
                help=f"Range: {min_val:.2f} - {max_val:.2f}"
            )
    
    if st.button("Make Prediction"):
        # Create input DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Check for warnings
        warning_flags_df, yellow_warnings, red_warnings, warning_ranges = check_input_values(input_df, training_data)
        
        # Scale the input
        scaled_input = scaler.transform(input_df)
        scaled_input_df = pd.DataFrame(scaled_input, columns=feature_names)
        
        # Make prediction
        prediction = predict(model, scaled_input_df)
        
        # Display results
        st.success(f"Prediction: **{prediction[0]:.2f} {target_column}**")
        
        # Display warnings if any
        display_input_warnings(yellow_warnings, red_warnings, warning_flags_df, warning_ranges, input_df)
        
        # Show input values used
        st.write("**Input values used:**")
        input_summary = pd.DataFrame({
            'Feature': list(input_data.keys()),
            'Value': list(input_data.values())
        })
        st.dataframe(input_summary, use_container_width=True)

    else:
        st.write("### Batch Prediction")
        st.write("Upload a file with the same format as the training data. The file should contain the following columns:")
        
        if training_data is not None:
            X_train = training_data['X_train']
            example_df = X_train.head(1)
            st.write("Required columns:")
            st.dataframe(example_df)
            
            st.write("Example data ranges:")
            ranges_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Min Value': X_train.min(),
                'Max Value': X_train.max()
            })
            st.dataframe(ranges_df)
        
        prediction_file = st.file_uploader("Upload data for batch prediction (CSV or Excel)", type=['csv', 'xlsx'], key="batch_file")
        if prediction_file is not None:
            try:
                pred_df = pd.read_csv(prediction_file) if prediction_file.name.lower().endswith('.csv') else pd.read_excel(prediction_file)
                
                # Validate columns
                if training_data is not None:
                    missing_cols = set(X_train.columns) - set(pred_df.columns)
                    extra_cols = set(pred_df.columns) - set(X_train.columns)
                    
                    if missing_cols:
                        st.error(f"Missing required columns: {', '.join(missing_cols)}")
                        st.stop()
                    
                    if extra_cols:
                        st.warning(f"Extra columns found (will be ignored): {', '.join(extra_cols)}")
                
                st.success("File loaded successfully!")
                st.write("Data Preview:")
                st.dataframe(pred_df.head())
                
                if st.button("Make Predictions", key="batch_predict"):
                    valid_features = [f for f in feature_names if f in pred_df.columns]
                    input_df = pred_df[valid_features].copy()
                    warning_flags_df, yellow_warnings, red_warnings, warning_ranges = check_input_values(input_df, training_data)
                    scaled_input_features = scaler.transform(input_df)
                    scaled_input_df = pd.DataFrame(scaled_input_features, columns=valid_features)
                    predictions = predict(model, scaled_input_df)
                    results_df = pred_df.copy()
                    results_df[f'Predicted {target_column}'] = predictions
                    results_df['Has Red Warning'] = warning_flags_df['has_red_warning']
                    results_df['Has Yellow Warning'] = warning_flags_df['has_yellow_warning']
                    results_df['Red Warning Features'] = warning_flags_df['red_warning_features']
                    results_df['Yellow Warning Features'] = warning_flags_df['yellow_warning_features']
                    st.success("Predictions completed!")
                    display_input_warnings(yellow_warnings, red_warnings, warning_flags_df, warning_ranges, input_df)
                    st.write("Prediction Results:")
                    st.dataframe(results_df)
                    csv = results_df.to_csv(index=False)
                    st.download_button("Download predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")
                    
                    # Add visualizations
                    st.subheader("Prediction Visualizations")
                    
                    # Create two columns for the plots
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Distribution plot
                        fig_dist = go.Figure()
                        fig_dist.add_trace(go.Histogram(
                            x=results_df[f'Predicted {target_column}'],
                            name='Distribution',
                            nbinsx=30
                        ))
                        fig_dist.update_layout(
                            title=f'Distribution of Predicted {target_column}',
                            xaxis_title=f'Predicted {target_column}',
                            yaxis_title='Count',
                            showlegend=False
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    with col2:
                        # Box plot
                        fig_box = go.Figure()
                        fig_box.add_trace(go.Box(
                            y=results_df[f'Predicted {target_column}'],
                            name='Predictions',
                            boxpoints='all'
                        ))
                        fig_box.update_layout(
                            title=f'Box Plot of Predicted {target_column}',
                            yaxis_title=f'Predicted {target_column}',
                            showlegend=False
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Time series plot if datetime column exists
                    datetime_cols = [col for col in results_df.columns if any(term in col.lower() for term in ['date', 'time', 'tarih'])]
                    if datetime_cols:
                        datetime_col = datetime_cols[0]  # Use the first datetime column found
                        try:
                            # Convert to datetime if not already
                            if not pd.api.types.is_datetime64_any_dtype(results_df[datetime_col]):
                                results_df[datetime_col] = pd.to_datetime(results_df[datetime_col])
                            
                            # Sort by datetime
                            results_df = results_df.sort_values(datetime_col)
                            
                            # Create time series plot
                            fig_ts = go.Figure()
                            fig_ts.add_trace(go.Scatter(
                                x=results_df[datetime_col],
                                y=results_df[f'Predicted {target_column}'],
                                mode='lines+markers',
                                name='Predictions'
                            ))
                            
                            # Add warning indicators if any
                            if 'Has Red Warning' in results_df.columns:
                                red_warning_points = results_df[results_df['Has Red Warning']]
                                if not red_warning_points.empty:
                                    fig_ts.add_trace(go.Scatter(
                                        x=red_warning_points[datetime_col],
                                        y=red_warning_points[f'Predicted {target_column}'],
                                        mode='markers',
                                        marker=dict(color='red', size=10, symbol='x'),
                                        name='Red Warnings'
                                    ))
                            
                            fig_ts.update_layout(
                                title=f'Predicted {target_column} Over Time',
                                xaxis_title='Time',
                                yaxis_title=f'Predicted {target_column}',
                                hovermode='x unified'
                            )
                            st.plotly_chart(fig_ts, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not create time series plot: {str(e)}")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        else:
            st.info("Please upload a file to make batch predictions.")

def create_scenario_dataframe(historical_df, years, feature_trends):
    """Creates a scenario dataframe by projecting the last known value forward."""
    freq = historical_df.index.freq or pd.infer_freq(historical_df.index)
    
    future_start = historical_df.index[-1] + freq
    future_end = future_start + pd.DateOffset(years=years) - freq
    future_dates = pd.date_range(start=future_start, end=future_end, freq=freq)
    periods = len(future_dates)
    
    future_df = pd.DataFrame(index=future_dates)
    periods_per_year = int(pd.Timedelta(days=365.25) / pd.to_timedelta(freq)) if freq else 365

    for feature in historical_df.columns:
        trend_mod = feature_trends.get(feature, {'type': 'Constant', 'add_seasonality': True})
        last_value = historical_df[feature].iloc[-1]
        future_values = np.full(periods, last_value, dtype=np.float64)

        if trend_mod['type'].lower() in ['linear', 'exponential', 'polynomial']:
            time_factor = np.linspace(0, years, periods)
            if trend_mod['type'].lower() == 'linear':
                future_values += time_factor * trend_mod['params']['slope']
            elif trend_mod['type'].lower() == 'exponential':
                future_values *= (1 + trend_mod['params']['growth_rate']) ** time_factor
            elif trend_mod['type'].lower() == 'polynomial':
                future_values += np.polyval(trend_mod['params']['coefficients'][::-1], time_factor)
        
        if trend_mod.get('add_seasonality', True):
            seasonal_periods = periods_per_year
            if len(historical_df[feature]) > 2 * seasonal_periods:
                try:
                    decomposition = seasonal_decompose(historical_df[feature], model='additive', period=seasonal_periods)
                    seasonal_values = decomposition.seasonal.iloc[-seasonal_periods:]
                    future_seasonal_values = np.tile(seasonal_values, int(np.ceil(periods / seasonal_periods)))[:periods]
                    future_values += future_seasonal_values
                except Exception as e:
                    st.warning(f"Could not determine seasonality for feature '{feature}': {e}. Proceeding without it.")
            else:
                st.warning(f"Not enough data for feature '{feature}' to determine seasonality (requires at least 2 full periods). Proceeding without it.")

        future_df[feature] = future_values

    return pd.concat([historical_df, future_df])

def apply_well_drilling_strategy(initial_future_features, initial_future_power, model, scaler, feature_names, threshold, muw_flowrate, steam_percentage, brine_col, steam_col, feature_trends):
    """Iteratively simulates well drilling and re-predicts power output."""
    progress_bar = st.progress(0, text="Simulation starting...")
    adjusted_features = initial_future_features.copy()
    adjusted_power = initial_future_power.copy()
    well_drilling_dates = []

    delta_ncg = muw_flowrate * (steam_percentage / 100.0)
    delta_brine = muw_flowrate * (1 - (steam_percentage / 100.0))
    
    year_starts = pd.date_range(start=adjusted_power.index.min(), end=adjusted_power.index.max(), freq='YS')
    num_years = len(year_starts)

    for i, start_of_year in enumerate(year_starts):
        progress_bar.progress((i + 1) / num_years, text=f"Simulating year {start_of_year.year}...")
        yearly_mask = (adjusted_power.index >= start_of_year) & (adjusted_power.index < start_of_year + pd.DateOffset(years=1))

        if yearly_mask.any() and adjusted_power[yearly_mask].mean() < threshold:
            drilling_date = start_of_year
            well_drilling_dates.append(drilling_date)
            
            future_mask = adjusted_features.index >= drilling_date
            affected_dates = adjusted_features.loc[future_mask].index
            years_from_drill = (affected_dates - drilling_date).days / 365.25

            # Brine decaying lift
            brine_trend = feature_trends.get(brine_col, {'type': 'constant'})
            brine_initial_lift = pd.Series(delta_brine, index=affected_dates)
            if brine_trend['type'].lower() == 'exponential':
                brine_decay_factor = (1 + brine_trend['params']['growth_rate']) ** years_from_drill
                brine_decayed_lift = brine_initial_lift * brine_decay_factor
            elif brine_trend['type'].lower() == 'linear':
                last_val = initial_future_features[brine_col].iloc[0]
                slope = brine_trend['params']['slope']
                initial_perc_decay = (slope / last_val) if last_val != 0 else 0
                brine_decay_factor = (1 + initial_perc_decay) ** years_from_drill
                brine_decayed_lift = brine_initial_lift * brine_decay_factor
            else:
                brine_decayed_lift = brine_initial_lift
            adjusted_features.loc[future_mask, brine_col] += brine_decayed_lift.clip(lower=0)

            # NCG/Steam decaying lift
            ncg_trend = feature_trends.get(steam_col, {'type': 'constant'})
            ncg_initial_lift = pd.Series(delta_ncg, index=affected_dates)
            if ncg_trend['type'].lower() == 'exponential':
                ncg_decay_factor = (1 + ncg_trend['params']['growth_rate']) ** years_from_drill
                ncg_decayed_lift = ncg_initial_lift * ncg_decay_factor
            elif ncg_trend['type'].lower() == 'linear':
                last_val = initial_future_features[steam_col].iloc[0]
                slope = ncg_trend['params']['slope']
                initial_perc_decay = (slope / last_val) if last_val != 0 else 0
                ncg_decay_factor = (1 + initial_perc_decay) ** years_from_drill
                ncg_decayed_lift = ncg_initial_lift * ncg_decay_factor
            else:
                ncg_decayed_lift = ncg_initial_lift
            adjusted_features.loc[future_mask, steam_col] += ncg_decayed_lift.clip(lower=0)

            # Re-predict power output
            X_scaled = scaler.transform(adjusted_features[feature_names])
            adjusted_power = pd.Series(model.predict(X_scaled), index=adjusted_features.index)

    progress_bar.progress(1.0, text="Simulation complete!")
    time.sleep(1)
    progress_bar.empty()
    return adjusted_power, well_drilling_dates, adjusted_features 