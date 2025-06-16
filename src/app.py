import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from predict import load_model, predict, check_input_values
from data_prep import load_data, preprocess_data
from time_series_analysis import (
    create_scenario_dataframe,
    plot_scenario,
    calculate_power_predictions,
    plot_power_predictions
)
import os
import json
from datetime import datetime
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from timeseries_cleaning_pipeline import (
    load_and_parse,
    enforce_frequency,
    sanity_checks,
    detect_and_impute_outliers
)

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
        # Display summary of warnings only for batch predictions
        total_rows = len(warning_flags_df)
        red_warning_rows = warning_flags_df['has_red_warning'].sum()
        yellow_warning_rows = warning_flags_df['has_yellow_warning'].sum()
        
        st.info(f"Warning Summary: {red_warning_rows} rows have red warnings, {yellow_warning_rows} rows have yellow warnings out of {total_rows} total rows.")

    if yellow_warnings or red_warnings:
        st.warning("⚠️ Current predictions are **UNRELIABLE**, see warnings.")

def load_latest_metrics(models_dir):
    """Load the latest metrics file from the models directory."""
    metrics_files = [f for f in os.listdir(models_dir) if f.startswith('metrics_') and f.endswith('.json')]
    if not metrics_files:
        return None
    
    # Sort by timestamp in filename and get the latest
    latest_metrics_file = sorted(metrics_files)[-1]
    metrics_path = os.path.join(models_dir, latest_metrics_file)
    
    with open(metrics_path, 'r') as f:
        return json.load(f)

def load_default_model(models_dir):
    """Load the default model files."""
    try:
        model = joblib.load(os.path.join(models_dir, "default_model.pkl"))
        scaler = joblib.load(os.path.join(models_dir, "default_scaler.pkl"))
        feature_names = joblib.load(os.path.join(models_dir, "default_feature_names.pkl"))
        return model, scaler, feature_names
    except Exception as e:
        print(f"Error loading default model files: {e}")
        return None, None, None

def load_latest_model_files(models_dir):
    """Load the latest model, scaler, and feature names files from the models directory."""
    # Get all files with their timestamps
    model_files = [f for f in os.listdir(models_dir) if f.startswith('xgboost_') and f.endswith('.pkl')]
    scaler_files = [f for f in os.listdir(models_dir) if f.startswith('scaler_') and f.endswith('.pkl')]
    feature_names_files = [f for f in os.listdir(models_dir) if f.startswith('feature_names_') and f.endswith('.pkl')]
    
    if not model_files or not scaler_files or not feature_names_files:
        return None, None, None
    
    # Get the latest files by sorting
    latest_model = sorted(model_files)[-1]
    latest_scaler = sorted(scaler_files)[-1]
    latest_feature_names = sorted(feature_names_files)[-1]
    
    # Load the files
    try:
        model = joblib.load(os.path.join(models_dir, latest_model))
        scaler = joblib.load(os.path.join(models_dir, latest_scaler))
        feature_names = joblib.load(os.path.join(models_dir, latest_feature_names))
        return model, scaler, feature_names
    except Exception as e:
        print(f"Error loading model files: {e}")
        return None, None, None

def load_training_data(models_dir):
    """Load the latest training data file from the models directory."""
    training_data_files = [f for f in os.listdir(models_dir) if f.startswith('training_data_') and f.endswith('.pkl')]
    if not training_data_files:
        return None
    
    # Sort by timestamp in filename and get the latest
    latest_training_data_file = sorted(training_data_files)[-1]
    training_data_path = os.path.join(models_dir, latest_training_data_file)
    
    try:
        return joblib.load(training_data_path)
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None

def display_data_visualizations(training_data, model=None):
    """Display various visualizations for the training data."""
    if training_data is None:
        return
    
    X_train = training_data['X_train']
    X_test = training_data['X_test']
    y_train = training_data['y_train']
    y_test = training_data['y_test']
    feature_names = training_data['feature_names']
    target_column = training_data.get('target_column', 'Target')
    
    # Combine train and test data for visualization
    X_combined = pd.concat([X_train, X_test])
    y_combined = pd.concat([y_train, y_test])
    
    # Create a DataFrame with all features and target
    df_combined = X_combined.copy()
    df_combined[target_column] = y_combined.values
    
    st.subheader("Training Data")
    
    # Feature importance plot
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
        st.plotly_chart(fig)

    # Correlation Heatmap
    corr = df_combined.corr()
    fig = px.imshow(corr, text_auto=True, title='Feature Correlation Heatmap')
    fig.update_layout(
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_nticks=len(corr.columns),
        yaxis_nticks=len(corr.index)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution plots for all features
    for col in df_combined.columns:
        fig = make_subplots(rows=1, cols=2, 
                          subplot_titles=(f'{col} Distribution', f'{col} Box Plot'))
        
        # Add histogram
        fig.add_trace(
            go.Histogram(x=df_combined[col], name='Histogram'),
            row=1, col=1
        )
        
        # Add box plot
        fig.add_trace(
            go.Box(y=df_combined[col], name='Box Plot'),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text=f'{col} Analysis'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_prediction_visualizations(results_df, target_column='Target'):
    """Display visualizations for batch prediction results."""
    st.subheader("Prediction Results Visualizations")
    
    # Distribution of predictions
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=('Prediction Distribution', 'Prediction Box Plot'))
    
    # Add histogram
    fig.add_trace(
        go.Histogram(x=results_df[f'Predicted {target_column}'], name='Histogram'),
        row=1, col=1
    )
    
    # Add box plot
    fig.add_trace(
        go.Box(y=results_df[f'Predicted {target_column}'], name='Box Plot'),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text=f'Predicted {target_column} Analysis'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Time series plot if there's a time column
    if 'Tarih' in results_df.columns:
        fig = px.line(results_df, x='Tarih', y=f'Predicted {target_column}',
                     title=f'Predicted {target_column} Over Time')
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation with input features
    st.write("Feature Correlations with Predictions")
    corr = results_df.corr()[f'Predicted {target_column}'].sort_values(ascending=False)
    fig = px.bar(x=corr.index, y=corr.values,
                title='Feature Correlations with Predictions',
                labels={'x': 'Features', 'y': 'Correlation'})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def cleanup_old_models(models_dir):
    """Clean up old model files, keeping only the default model and the latest trained model."""
    # Keep track of files to preserve
    files_to_keep = set([
        "default_model.pkl",
        "default_scaler.pkl",
        "default_feature_names.pkl",
        "default_metrics.json",
        "default_training_data.pkl"
    ])
    
    # Get all model-related files
    all_files = os.listdir(models_dir)
    
    # Find the latest model files
    model_files = [f for f in all_files if f.startswith('xgboost_') and f.endswith('.pkl')]
    scaler_files = [f for f in all_files if f.startswith('scaler_') and f.endswith('.pkl')]
    feature_names_files = [f for f in all_files if f.startswith('feature_names_') and f.endswith('.pkl')]
    metrics_files = [f for f in all_files if f.startswith('metrics_') and f.endswith('.json')]
    training_data_files = [f for f in all_files if f.startswith('training_data_') and f.endswith('.pkl')]
    
    if model_files:
        files_to_keep.add(sorted(model_files)[-1])
    if scaler_files:
        files_to_keep.add(sorted(scaler_files)[-1])
    if feature_names_files:
        files_to_keep.add(sorted(feature_names_files)[-1])
    if metrics_files:
        files_to_keep.add(sorted(metrics_files)[-1])
    if training_data_files:
        files_to_keep.add(sorted(training_data_files)[-1])
    
    # Remove old files
    for file in all_files:
        if file not in files_to_keep:
            try:
                os.remove(os.path.join(models_dir, file))
            except Exception as e:
                print(f"Error removing file {file}: {e}")

def load_selected_model_components(model_option, models_dir):
    """Loads the appropriate model components based on user selection."""
    model, scaler, feature_names, training_data, status = None, None, None, None, ""

    if model_option == "Use Default Model":
        model, scaler, feature_names = load_default_model(models_dir)
        if model and scaler and feature_names:
            try:
                training_data = joblib.load(os.path.join(models_dir, "default_training_data.pkl"))
                status = "Default model loaded successfully."
            except Exception as e:
                status = f"Default model loaded, but error loading training data: {e}"
        else:
            status = "Default model not found. Please train a new model."

    elif model_option == "Train New Model":
        # When training a new model, the components are stored in session state after training
        if st.session_state.get('new_model_trained', False):
            model = st.session_state.get('new_model_model')
            scaler = st.session_state.get('new_model_scaler')
            feature_names = st.session_state.get('new_model_feature_names')
            training_data = st.session_state.get('new_model_training_data')
            if model and scaler and feature_names and training_data:
                 status = "Newly trained model loaded successfully."
            else:
                 status = "No new model has been trained yet."
        else:
            status = "No new model has been trained yet."

    return model, scaler, feature_names, training_data, status

def handle_prediction_workflow(model, scaler, feature_names, training_data):
    """Handles the prediction input, processing, and output display."""
    st.subheader("Make Predictions")
    input_method = st.radio(
        "Select input method:",
        ["Single Prediction", "Batch Prediction"],
        key=f"{model.n_estimators if hasattr(model, 'n_estimators') else 'default'}_predict_method"
    )

    # Get target column name from training data
    target_column = training_data.get('target_column', 'Target')

    if training_data is not None:
        X_train = training_data['X_train']
        # Ensure feature_names is in sync with X_train
        feature_names = [f for f in feature_names if f in X_train.columns]

    if input_method == "Single Prediction":
        st.write("Enter values for prediction:")
        input_data = {}

        if training_data is not None:
            for feature in feature_names:
                if feature not in X_train.columns:
                    continue
                train_q1 = X_train[feature].quantile(0.25)
                train_q3 = X_train[feature].quantile(0.75)
                train_iqr = train_q3 - train_q1
                iqr_lower = train_q1 - 1.5 * train_iqr
                iqr_upper = train_q3 + 1.5 * train_iqr

                input_data[feature] = st.number_input(
                    f"{feature} (IQR Range: {iqr_lower:.2f} - {iqr_upper:.2f})",
                    value=0.0,
                    format="%.4f",
                    key=f"{model.n_estimators if hasattr(model, 'n_estimators') else 'default'}_input_{feature}"
                )
        else:
            st.warning("Training data not available. Cannot display typical ranges.")
            for feature in feature_names:
                input_data[feature] = st.number_input(
                    f"{feature}",
                    value=0.0,
                    format="%.4f",
                    key=f"{model.n_estimators if hasattr(model, 'n_estimators') else 'default'}_input_{feature}"
                )

        if st.button("Predict", key=f"{model.n_estimators if hasattr(model, 'n_estimators') else 'default'}_single_predict"):
            try:
                input_df = pd.DataFrame([input_data])

                warning_flags_df, yellow_warnings, red_warnings, warning_ranges = check_input_values(input_df, training_data)

                scaled_input_features = scaler.transform(input_df)
                scaled_input_df = pd.DataFrame(scaled_input_features, columns=feature_names)
                prediction_value = predict(model, scaled_input_df)

                display_input_warnings(yellow_warnings, red_warnings, warning_flags_df, warning_ranges, input_df)

                st.success(f"Predicted {target_column}: {prediction_value[0]:.2f}")

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    else:  # Batch Prediction
        prediction_file = st.file_uploader("Upload data for batch prediction (CSV or Excel)", type=['csv', 'xlsx'], key=f"{model.n_estimators if hasattr(model, 'n_estimators') else 'default'}_batch_file")

        if prediction_file is not None:
            try:
                pred_df = load_data(prediction_file)
                st.success("File loaded successfully!")
                st.write("Data Preview:")
                st.dataframe(pred_df.head())

                if st.button("Make Predictions", key=f"{model.n_estimators if hasattr(model, 'n_estimators') else 'default'}_batch_predict"):
                    try:
                        # Only use features present in both feature_names and pred_df
                        valid_features = [f for f in feature_names if f in pred_df.columns]
                        input_df = pred_df[valid_features].copy()

                        warning_flags_df, yellow_warnings, red_warnings, warning_ranges = check_input_values(input_df, training_data)

                        scaled_input_features = scaler.transform(input_df)
                        scaled_input_df = pd.DataFrame(scaled_input_features, columns=valid_features)
                        predictions = predict(model, scaled_input_df)
                        results_df = pred_df.copy()
                        results_df[f'Predicted {target_column}'] = predictions

                        # Add warning flags to results
                        results_df['Has Red Warning'] = warning_flags_df['has_red_warning']
                        results_df['Has Yellow Warning'] = warning_flags_df['has_yellow_warning']
                        results_df['Red Warning Features'] = warning_flags_df['red_warning_features']
                        results_df['Yellow Warning Features'] = warning_flags_df['yellow_warning_features']

                        st.success("Predictions completed!")

                        display_input_warnings(yellow_warnings, red_warnings, warning_flags_df, warning_ranges, input_df)

                        st.write("Prediction Results:")
                        st.dataframe(results_df)

                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download predictions as CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Error making predictions: {str(e)}")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
            else:
                st.info("Please upload a file to make batch predictions.")

def display_model_metrics(metrics):
    """Displays model metrics and visualizations in the sidebar."""
    if metrics:
        st.sidebar.title("Model Information")
        st.sidebar.write(f"Model Type: {metrics.get('model_type', 'N/A')}")
        timestamp_str = metrics.get('timestamp')
        if timestamp_str:
            st.sidebar.write(f"Training Date: {datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')}")

        st.sidebar.subheader("Performance Metrics")

        test_metrics = metrics.get('metrics', {}).get('test_metrics', {})
        if test_metrics:
             st.sidebar.write("Test Set Performance:")
             st.sidebar.write(f"R² Score: {test_metrics.get('r2', 0.0):.4f}")
             st.sidebar.write(f"RMSE: {test_metrics.get('rmse', 0.0):.4f}")
             st.sidebar.write(f"MSE: {test_metrics.get('mse', 0.0):.4f}")
        else:
             # For backward compatibility with older metrics structure
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

            with st.sidebar.expander("Individual CV Scores"):
                r2_scores = cv_metrics.get('cv_scores', {}).get('r2', [])
                if r2_scores:
                     st.write("R² Scores per fold:")
                     for i, score in enumerate(r2_scores, 1):
                        st.write(f"Fold {i}: {score:.4f}")
                rmse_scores = cv_metrics.get('cv_scores', {}).get('rmse', [])
                if rmse_scores:
                     st.write("RMSE Scores per fold:")
                     for i, score in enumerate(rmse_scores, 1):
                        st.write(f"Fold {i}: {score:.4f}")

        with st.sidebar.expander("Model Parameters"):
            model_params = metrics.get('model_params', {})
            if model_params:
                for param, value in model_params.items():
                    st.write(f"{param}: {value}")
            else:
                st.write("No model parameters found.")

        actual = metrics.get('actual')
        predicted = metrics.get('predicted')
        if actual and predicted:
             df_plot = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
             fig = px.scatter(df_plot, x='Actual', y='Predicted',
                              title='Actual vs Predicted',
                              labels={'Actual': 'Actual Brüt Güç', 'Predicted': 'Predicted Brüt Güç'})
             min_val = min(min(actual), min(predicted))
             max_val = max(max(actual), max(predicted))
             fig.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                           line=dict(color='black', dash='dash'))
             st.sidebar.plotly_chart(fig, use_container_width=True)

        loss_curve = metrics.get('metrics', {}).get('loss_curve', {})
        if loss_curve and loss_curve.get('train_loss') and loss_curve.get('test_loss'):
            epochs = range(1, len(loss_curve['train_loss']) + 1)
            df_loss = pd.DataFrame({
                'Epoch': epochs,
                'Training Loss': loss_curve['train_loss'],
                'Validation Loss': loss_curve['test_loss']
            })
            fig = px.line(df_loss, x='Epoch', y=['Training Loss', 'Validation Loss'],
                        title='Training and Validation Loss',
                        labels={'value': 'RMSE Loss', 'variable': 'Dataset'})
            st.sidebar.plotly_chart(fig, use_container_width=True)

    else:
        st.sidebar.warning("No model metrics found.")

def create_scenario_dataframe(historical_df, years, feature_trends):
    """
    Creates a scenario dataframe.
    Projects the last known value forward, optionally applying a trend and/or seasonality.
    """
    freq = historical_df.index.freq
    if freq is None:
        freq = pd.infer_freq(historical_df.index)

    future_start = historical_df.index[-1] + freq
    future_end = future_start + pd.DateOffset(years=years) - freq
    future_dates = pd.date_range(start=future_start, end=future_end, freq=freq)
    periods = len(future_dates)

    future_df = pd.DataFrame(index=future_dates)
    
    # Estimate periods per year for seasonality decomposition (this can be an approximation)
    periods_per_year = int(pd.Timedelta(days=365.25) / pd.to_timedelta(freq)) if freq else 365

    for feature in historical_df.columns:
        # Get trend settings
        trend_mod = feature_trends.get(feature, {'type': 'Constant', 'add_seasonality': True})
        
        # Start with the last known value as a constant baseline
        last_value = historical_df[feature].iloc[-1]
        future_values = np.full(periods, last_value, dtype=np.float64)

        # Apply user-defined trend modifications if specified
        if trend_mod['type'] in ['linear', 'exponential', 'polynomial']:
            time_factor = np.linspace(0, years, periods)  # Time in years from start

            if trend_mod['type'] == 'linear':
                # Absolute change per year
                future_values += time_factor * trend_mod['params']['slope']
            elif trend_mod['type'] == 'exponential':
                # Percentage change per year
                future_values *= (1 + trend_mod['params']['growth_rate']) ** time_factor
            elif trend_mod['type'] == 'polynomial':
                # Additive polynomial trend
                poly_trend = np.polyval(trend_mod['params']['coefficients'][::-1], time_factor)
                future_values += poly_trend
        
        # Add seasonality, if selected
        if trend_mod.get('add_seasonality', True): # Default to True for safety
            seasonal_periods = periods_per_year
            if len(historical_df[feature]) > 2 * seasonal_periods:
                decomposition = seasonal_decompose(historical_df[feature], model='additive', period=seasonal_periods)
                
                # Extrapolate Seasonality by tiling
                seasonal_values = decomposition.seasonal.iloc[-seasonal_periods:]
                future_seasonal_values = np.tile(seasonal_values, int(np.ceil(periods / seasonal_periods)))[:periods]
                future_values += future_seasonal_values
            else:
                st.warning(f"Not enough data for feature '{feature}' to determine seasonality. Projecting without seasonality.")

        future_df[feature] = future_values

    # Combine historical and future data
    scenario_df = pd.concat([historical_df, future_df])
    
    return scenario_df

def plot_scenario(scenario_data, years, target_col=None, feature_trends=None):
    """
    Plot the scenario data with separate subplots for each feature.
    The target plot is emphasized with a taller y-axis and a clear label.
    Modifiers are shown above the plots.
    """
    # Downsample data for plotting to improve performance
    if len(scenario_data) > 2000:  # Only sample if data is large
        plot_data = scenario_data.resample('D').mean()
    else:
        plot_data = scenario_data

    # Show modifiers summary
    if feature_trends:
        st.markdown("**Applied Modifiers:**")
        for feature, trend in feature_trends.items():
            if trend['type'] == 'linear':
                st.write(f"- {feature}: Linear ({trend['params']['slope']*100:.2f} per year)")
            elif trend['type'] == 'exponential':
                st.write(f"- {feature}: Exponential ({trend['params']['growth_rate']*100:.2f}% per year)")
            elif trend['type'] == 'polynomial':
                st.write(f"- {feature}: Polynomial (coefficients: {trend['params']['coefficients']})")
            else:
                st.write(f"- {feature}: Constant")

    # Define color pairs for features
    color_pairs = [
        ('blue', 'lightblue'),
        ('red', 'lightcoral'),
        ('green', 'lightgreen'),
        ('purple', 'plum'),
        ('orange', 'peachpuff'),
        ('brown', 'burlywood'),
        ('pink', 'lightpink'),
        ('cyan', 'lightcyan'),
        ('magenta', 'lavender'),
        ('olive', 'beige')
    ]

    # Separate target from features
    features = [col for col in plot_data.columns if col != target_col]
    n_features = len(features)
    total_rows = n_features + 1  # +1 for target

    # Create subplots: target plot is first and taller
    row_heights = [0.5] + [0.5 / n_features] * n_features  # Target plot is 50% height, others share the rest
    fig = make_subplots(
        rows=total_rows,
        cols=1,
        subplot_titles=[f"Predicted {target_col}"] + features,
        row_heights=row_heights,
        vertical_spacing=0.08
    )

    # Get the split point between historical and future data
    if len(plot_data) > 0:
        split_date = plot_data.index[-int(years * 365.25 * (plot_data.index.freq / pd.Timedelta(days=1)))]
    else:
        split_date = pd.Timestamp.now()

    # Plot target (emphasized)
    actual_color, future_color = color_pairs[0]
    historical_target = plot_data[target_col][:split_date]
    future_target = plot_data[target_col][split_date:]
    fig.add_trace(
        go.Scatter(
            x=historical_target.index,
            y=historical_target.values,
            name=f'{target_col} (Historical)',
            line=dict(color=actual_color, width=3)
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=future_target.index,
            y=future_target.values,
            name=f'{target_col} (Predicted)',
            line=dict(color=future_color, dash='dash', width=3)
        ),
        row=1,
        col=1
    )

    # Plot each feature in its own subplot
    for i, feature in enumerate(features, 2):
        actual_color, future_color = color_pairs[i % len(color_pairs)]
        historical_data = plot_data[feature][:split_date]
        future_data = plot_data[feature][split_date:]
        fig.add_trace(
            go.Scatter(
                x=historical_data.index,
                y=historical_data.values,
                name=f'{feature} (Historical)',
                line=dict(color=actual_color)
            ),
            row=i,
            col=1
        )
        fig.add_trace(
            go.Scatter(
                x=future_data.index,
                y=future_data.values,
                name=f'{feature} (Projected)',
                line=dict(color=future_color, dash='dash')
            ),
            row=i,
            col=1
        )

    # Update layout
    fig.update_layout(
        title=f'Feature Trends and {target_col} Prediction',
        height=400 + 200 * n_features,  # Target plot is taller
        showlegend=True,
        hovermode='x unified'
    )
    # Emphasize y-axis for target
    fig.update_yaxes(title_text=target_col, row=1, col=1)
    for i, feature in enumerate(features, 2):
        fig.update_yaxes(title_text=feature, row=i, col=1)

    return fig

def main():
    # Set page configuration
    st.set_page_config(
        page_title="GPOP",
        page_icon="💨",
        layout="wide"
    )

    st.title("Geothermal Power Output Prediction")

    # Add tabs for different functionalities
    tab1, tab2 = st.tabs(["Model Training & Prediction", "Time Series Forecasting"])

    with tab1:
        # Add model selection in sidebar
        st.sidebar.title("Model Options")
        model_option = st.sidebar.radio(
            "Choose an option:",
            ["Use Default Model", "Train New Model"]
        )

        models_dir = "models"

        if model_option == "Train New Model":
            st.write("Upload your data to train a new model.")
            training_file = st.file_uploader("Upload training data (CSV or Excel)", type=['csv', 'xlsx'], key="training_file")

            if training_file is not None:
                try:
                    # Use a temporary path for the uploaded file
                    with open(training_file.name, "wb") as f:
                        f.write(training_file.getbuffer())
                    
                    # --- New Cleaning Pipeline Integration ---
                    st.subheader("Time Series Settings for Training Data")
                    
                    # Let user pick datetime column
                    if training_file.name.lower().endswith('.csv'):
                        temp_df = pd.read_csv(training_file.name, nrows=0) # Read only headers
                    else:
                        temp_df = pd.read_excel(training_file.name, nrows=0) # Read only headers
                    
                    datetime_col = st.selectbox(
                        "Select your datetime column",
                        temp_df.columns.tolist()
                    )
                    
                    # Let user pick frequency
                    frequency = st.selectbox(
                        "Select data frequency",
                        ["1min", "5min", "15min", "30min", "1H", "2H", "4H", "6H", "8H", "12H", "1D"],
                        index=4,
                        key="train_frequency"
                    )

                    # Load and clean data using the new pipeline
                    df = load_and_parse(training_file.name, datetime_col=datetime_col)
                    df = enforce_frequency(df, freq=frequency)
                    sanity_checks(df)
                    
                    st.success("File loaded and preprocessed successfully!")
                    
                    # --- End of New Cleaning Pipeline Integration ---
                    
                    st.write("Data Preview:")
                    st.dataframe(df.head())

                    st.write("Dataset Information:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Number of rows: {len(df)}")
                        st.write(f"Number of features: {len(df.columns)}")
                    with col2:
                        st.write("Feature names:")
                        for col in df.columns:
                            st.write(f"- {col}")

                    st.write("Basic Statistics:")
                    st.dataframe(df.describe())

                    # User picks target and flowrate columns
                    st.subheader("Column Mapping for Model Training & Simulation")
                    all_cols = df.columns.tolist()
                    
                    target_col = st.selectbox(
                        "Select Target Column (e.g., Gross Power)",
                        all_cols,
                        index=len(all_cols) - 1,
                        key="train_target_col"
                    )

                    # Let user map brine and steam flowrate columns for the simulation
                    st.write("Select the columns that represent your primary brine and steam flowrates. This is used for the new well simulation.")
                    
                    available_cols = [col for col in all_cols if col != target_col]
                    
                    brine_col = st.selectbox(
                        "Select Primary Brine Flowrate Column",
                        available_cols,
                        key="train_brine_col"
                    )
                    
                    steam_col = st.selectbox(
                        "Select Primary Steam Flowrate Column",
                        [col for col in available_cols if col != brine_col],
                        key="train_steam_col"
                    )

                    # Outlier detection on numeric features (excluding target)
                    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                    if target_col in numeric_cols:
                        numeric_cols.remove(target_col)
                    
                    df = detect_and_impute_outliers(df, cols=numeric_cols)
                    st.info("Outlier detection and imputation complete.")

                    training_data_for_viz = {
                        'X_train': df.drop(target_col, axis=1),
                        'X_test': df.drop(target_col, axis=1).iloc[:len(df)//5],
                        'y_train': df[target_col],
                        'y_test': df[target_col].iloc[:len(df)//5],
                        'feature_names': df.drop(target_col, axis=1).columns.tolist(),
                        'target_column': target_col,
                        'brine_col': brine_col,
                        'steam_col': steam_col
                    }

                    # Add hyperparameter tuning section
                    st.subheader("Model Hyperparameters")
                    st.write("Adjust the model hyperparameters below (optional):")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        n_estimators = st.slider("Number of Trees (n_estimators)", 
                                              min_value=50, max_value=500, 
                                              value=300, step=50)
                        learning_rate = st.slider("Learning Rate", 
                                               min_value=0.01, max_value=0.3, 
                                               value=0.05, step=0.01)
                        max_depth = st.slider("Maximum Tree Depth", 
                                           min_value=3, max_value=10, 
                                           value=4, step=1)
                    
                    with col2:
                        min_child_weight = st.slider("Minimum Child Weight", 
                                                  min_value=1, max_value=10, 
                                                  value=4, step=1)
                        subsample = st.slider("Subsample Ratio", 
                                           min_value=0.5, max_value=1.0, 
                                           value=0.8, step=0.1)
                        colsample_bytree = st.slider("Column Sample by Tree", 
                                                  min_value=0.5, max_value=1.0, 
                                                  value=0.8, step=0.1)

                    # Store hyperparameters in session state
                    st.session_state['model_params'] = {
                        'n_estimators': n_estimators,
                        'learning_rate': learning_rate,
                        'max_depth': max_depth,
                        'min_child_weight': min_child_weight,
                        'subsample': subsample,
                        'colsample_bytree': colsample_bytree
                    }

                    # Add a slider for test_size (train/test split ratio)
                    test_size = st.slider("Test Set Size (Fraction)", min_value=0.1, max_value=0.5, value=0.2, step=0.01, help="Fraction of data to use as test set (e.g., 0.2 = 20%)")

                    if 'new_model_metrics' not in st.session_state:
                        st.session_state['new_model_metrics'] = None
                    if 'new_model_trained' not in st.session_state:
                        st.session_state['new_model_trained'] = False

                    if st.button("Train New Model"):
                        try:
                            from train import train_model
                            # Get custom hyperparameters from session state if available
                            model_params = st.session_state.get('model_params')
                            metrics = train_model(training_file, models_dir, target_column=target_col, model_params=model_params, test_size=test_size)
                            model, scaler, feature_names = load_latest_model_files(models_dir)
                            st.success("New model trained and loaded successfully!")
                            cleanup_old_models(models_dir)
                            st.session_state['new_model_metrics'] = metrics
                            st.session_state['new_model_trained'] = True
                            st.session_state['new_model_training_data'] = training_data_for_viz # Store for prediction use
                            st.session_state['new_model_scaler'] = scaler
                            st.session_state['new_model_model'] = model
                            st.session_state['new_model_feature_names'] = feature_names
                        except Exception as e:
                            st.error(f"Error training model: {str(e)}")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                finally:
                    # Clean up the temporary file
                    if os.path.exists(training_file.name):
                        os.remove(training_file.name)
            else:
                st.info("Please upload a training data file to train a new model.")

        # Load the selected model components
        model, scaler, feature_names, training_data, status = load_selected_model_components(model_option, models_dir)

        st.sidebar.write(status)

        # Always get feature_names and target_column from training_data
        if training_data is not None:
            feature_names = training_data.get('feature_names', feature_names)
            target_col = training_data.get('target_column', None)

        if model and scaler and feature_names and training_data:
            
            explore_tab, predict_tab = st.tabs(["Explore Model", "Make Predictions"])

            with explore_tab:
                st.header("Model & Training Data Exploration")
                # Display training data visualizations
                display_data_visualizations(training_data, model)

            with predict_tab:
                st.header("Make Predictions with the Loaded Model")
                # Handle prediction workflow
                handle_prediction_workflow(model, scaler, feature_names, training_data)

            # Load and display metrics in sidebar
            metrics = None
            if model_option == "Use Default Model":
                metrics_path = os.path.join(models_dir, "default_metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
            elif model_option == "Train New Model" and st.session_state.get('new_model_trained', False):
                metrics = st.session_state.get('new_model_metrics')

            display_model_metrics(metrics)

    with tab2:
        # Only show time series analysis if a model is available
        show_time_series = True
        if model_option == "Train New Model" and not st.session_state.get('new_model_trained', False):
            show_time_series = False
            st.warning("Please train a new model first to use Time Series.")
        if show_time_series:
            st.title("Time Series Forecasting")

            # Use the loaded training_data and model for time series
            try:
                if model_option == "Use Default Model":
                    # For default model, load default data and model using the new pipeline
                    peek_df = pd.read_excel("data/default_data.xlsx", nrows=0)
                    datetime_col = peek_df.columns[0]
                    default_data = load_and_parse("data/default_data.xlsx", datetime_col=datetime_col)
                    default_data = enforce_frequency(default_data, freq='H')
                    sanity_checks(default_data)
                    
                    model, scaler, feature_names = load_default_model("models")
                    # Get target column from default model's training data
                    target_col = joblib.load(os.path.join("models", "default_training_data.pkl")).get('target_column', default_data.columns[-1])
                    brine_col = "Brine Flowrate (T/h)"
                    steam_col = "NCG+Steam Flowrate (T/h)"
                    ts_data = default_data
                else:
                    # For new model, use the latest training data and model
                    ts_data = training_data['X_train'].copy()
                    ts_data[training_data['target_column']] = training_data['y_train']
                    model = st.session_state.get('new_model_model')
                    scaler = st.session_state.get('new_model_scaler')
                    feature_names = st.session_state.get('new_model_feature_names')
                    target_col = training_data['target_column']
                    brine_col = training_data.get('brine_col')
                    steam_col = training_data.get('steam_col')

                st.write("Training Data Preview:")
                st.dataframe(ts_data.head())

                # Use all features except the target column
                selected_features = [col for col in ts_data.columns if col != target_col]

                if selected_features:
                    # Configure trends
                    st.subheader("Feature Trends")
                    years = st.slider("Number of years to project", 1, 50, 20)

                    # --- Default trend settings ---
                    default_trends = {
                        brine_col: {"type": "Exponential", "value": -3.0, "seasonality": True},
                        steam_col: {"type": "Exponential", "value": -3.0, "seasonality": True},
                        "Ambient Temperature (°C)": {"type": "Linear", "value": 1.0, "seasonality": True},
                        "Heat Exchanger Pressure Differential (Bar)": {"type": "Constant", "value": 0.0, "seasonality": False},
                    }

                    feature_trends = {}
                    for feature in selected_features:
                        st.write(f"### {feature}")

                        # Get default settings for the current feature
                        defaults = default_trends.get(feature, {"type": "Constant", "value": 0.0, "seasonality": True})
                        trend_options = ["Constant", "Linear", "Exponential", "Polynomial"]
                        default_index = trend_options.index(defaults["type"]) if defaults["type"] in trend_options else 0

                        col1, col2 = st.columns(2)
                        with col1:
                            trend_type = st.selectbox(
                                "Select Trend Type",
                                trend_options,
                                index=default_index,
                                key=f"trend_{feature}"
                            )
                        with col2:
                            add_seasonality = st.checkbox(
                                "Seasonality",
                                value=defaults.get("seasonality", True),
                                key=f"seasonality_{feature}"
                            )

                        feature_trends[feature] = {
                            'type': trend_type,
                            'add_seasonality': add_seasonality
                        }
                        
                        if trend_type == "Linear":
                            slope = st.number_input(
                                f"Annual change for {feature}",
                                value=defaults["value"] if defaults["type"] == "Linear" else 0.0,
                                format="%.2f",
                                key=f"slope_{feature}"
                            )
                            feature_trends[feature]['params'] = {'slope': slope}
                        elif trend_type == "Exponential":
                            growth_rate = st.number_input(
                                f"Annual growth rate for {feature} (%)",
                                value=defaults["value"] if defaults["type"] == "Exponential" else 0.0,
                                format="%.2f",
                                key=f"growth_{feature}"
                            )
                            feature_trends[feature]['params'] = {'growth_rate': growth_rate / 100}
                        elif trend_type == "Polynomial":
                            degree = st.slider(
                                f"Polynomial degree for {feature}",
                                1, 5,
                                key=f"degree_{feature}"
                            )
                            coefficients = []
                            for i in range(degree + 1):
                                coef = st.number_input(
                                    f"Coefficient for x^{i}",
                                    value=0.0,
                                    format="%.2f",
                                    key=f"coef_{feature}_{i}"
                                )
                                coefficients.append(coef)
                            feature_trends[feature]['params'] = {'coefficients': coefficients}

                    if st.button("Generate Scenario"):
                        # Create scenario dataframe with extrapolated features (exclude target)
                        scenario_features = create_scenario_dataframe(ts_data[selected_features], years, feature_trends)
                        scenario_data = scenario_features.copy()

                        if model and scaler:
                            # Prepare data for prediction
                            X_all = scenario_data[selected_features]
                            X_scaled = scaler.transform(X_all)
                            predictions = model.predict(X_scaled)

                            # For historical part, use original target; for future, use predictions
                            n_hist = len(ts_data)
                            scenario_data[target_col] = np.concatenate([
                                ts_data[target_col].values,
                                predictions[n_hist:]
                            ])
                            
                            # Round data and store in session state
                            scenario_data = scenario_data.round(4)
                            st.session_state.scenario_data = scenario_data
                            st.session_state.years = years
                            st.session_state.target_col = target_col
                            st.session_state.feature_trends = feature_trends

                            # Also prepare and store the CSV data
                            st.session_state.csv_no_sim = scenario_data.to_csv()

                        else:
                            st.error("Failed to load the model. Please ensure the model files exist.")
                    
                    if 'scenario_data' in st.session_state:
                        # Retrieve data from session state
                        scenario_data = st.session_state.scenario_data
                        years = st.session_state.years
                        target_col = st.session_state.target_col
                        feature_trends = st.session_state.feature_trends
                        
                        # Display results
                        st.subheader(f"Feature Trends and {target_col} Predictions")

                        if len(scenario_data) > 2000:
                            st.info("ℹ️ To improve performance, charts are downsampled to show daily averages. The downloaded data contains the full, original-resolution data.")

                        st.plotly_chart(plot_scenario(scenario_data, years, target_col=target_col, feature_trends=feature_trends), use_container_width=True)

                        # Download button for data without well simulation
                        if 'csv_no_sim' in st.session_state:
                            st.download_button(
                                label="Download Scenario Data (no MUW simulation)",
                                data=st.session_state.csv_no_sim,
                                file_name="scenario_predictions_no_simulation.csv",
                                mime="text/csv",
                                key="download_no_sim"
                            )

                        # Calculate and display yearly averages of projected predictions (future)
                        split_date = scenario_data.index[-1] - pd.DateOffset(years=years)
                        future_data = scenario_data[scenario_data.index > split_date]

                        # ------------------ Well Drilling Simulation ------------------
                        st.subheader("Well Drilling Simulation")
                        threshold = st.number_input(
                            "Yearly average power threshold for drilling a new well (MW)",
                            min_value=0,
                            value=40,
                            step=1,
                            key="well_threshold"
                        )
                        muw_flowrate = st.number_input(
                            "Make-up Well Flowrate (T/h)",
                            min_value=0.0,
                            value=400.0,
                            step=10.0,
                            key="muw_flowrate"
                        )
                        steam_percentage = st.number_input(
                            "Make-up Well Steam Percentage (%)",
                            min_value=0.0,
                            max_value=100.0,
                            value=10.0,
                            step=1.0,
                            key="steam_percentage"
                        )

                        if st.button("Simulate New Wells"):
                            progress_placeholder = st.empty()
                            
                            def apply_well_drilling_strategy(
                                initial_future_features,
                                initial_future_power,
                                model,
                                scaler,
                                feature_names,
                                threshold,
                                muw_flowrate,
                                steam_percentage,
                                brine_col,
                                steam_col,
                                feature_trends
                            ):
                                """
                                Iteratively simulates well drilling. When yearly average power drops below
                                the threshold, a new well's contribution is added to the input features,
                                and the future power output is re-predicted. The new well's contribution
                                also decays over time according to the specified feature trends.
                                """
                                adjusted_features = initial_future_features.copy()
                                adjusted_power = initial_future_power.copy()
                                well_drilling_dates = []

                                delta_ncg = muw_flowrate * (steam_percentage / 100.0)
                                delta_brine = muw_flowrate * (1 - (steam_percentage / 100.0))

                                year_starts = pd.date_range(
                                    start=adjusted_power.index.min(),
                                    end=adjusted_power.index.max(),
                                    freq='YS'
                                )

                                num_years = len(year_starts)
                                progress_bar = progress_placeholder.progress(0, text="Simulation starting...")

                                for i, start_of_year in enumerate(year_starts):
                                    end_of_year = start_of_year + pd.DateOffset(years=1)
                                    yearly_mask = (adjusted_power.index >= start_of_year) & (adjusted_power.index < end_of_year)

                                    if yearly_mask.any():
                                        yearly_average = adjusted_power[yearly_mask].mean()

                                        if yearly_average < threshold:
                                            drilling_date = start_of_year
                                            well_drilling_dates.append(drilling_date)
                                            
                                            future_mask = adjusted_features.index >= drilling_date
                                            affected_dates = adjusted_features.loc[future_mask].index
                                            years_from_drill = (affected_dates - drilling_date).days / 365.25

                                            # --- Calculate decaying lift for Brine ---
                                            brine_trend = feature_trends.get(brine_col, {'type': 'constant'})
                                            brine_initial_lift = pd.Series(delta_brine, index=affected_dates)
                                            
                                            if brine_trend['type'] == 'exponential':
                                                growth_rate = brine_trend['params']['growth_rate']
                                                brine_decayed_lift = brine_initial_lift * ((1 + growth_rate) ** years_from_drill)
                                            elif brine_trend['type'] == 'linear':
                                                # The slope is the absolute change per year.
                                                slope = brine_trend['params']['slope']
                                                # The lift from the well also decays by the same absolute amount.
                                                total_change = slope * years_from_drill
                                                brine_decayed_lift = brine_initial_lift + total_change
                                            else: # Constant
                                                brine_decayed_lift = brine_initial_lift
                                            brine_decayed_lift = brine_decayed_lift.clip(lower=0)
                                            
                                            # --- Calculate decaying lift for NCG+Steam ---
                                            ncg_trend = feature_trends.get(steam_col, {'type': 'constant'})
                                            ncg_initial_lift = pd.Series(delta_ncg, index=affected_dates)
                                            
                                            if ncg_trend['type'] == 'exponential':
                                                growth_rate = ncg_trend['params']['growth_rate']
                                                ncg_decayed_lift = ncg_initial_lift * ((1 + growth_rate) ** years_from_drill)
                                            elif ncg_trend['type'] == 'linear':
                                                slope = ncg_trend['params']['slope']
                                                total_change = slope * years_from_drill
                                                ncg_decayed_lift = ncg_initial_lift + total_change
                                            else: # Constant
                                                ncg_decayed_lift = ncg_initial_lift
                                            ncg_decayed_lift = ncg_decayed_lift.clip(lower=0)

                                            # Add the decaying lifts
                                            adjusted_features.loc[future_mask, brine_col] += brine_decayed_lift
                                            adjusted_features.loc[future_mask, steam_col] += ncg_decayed_lift

                                            # With the features adjusted, re-predict the power output for all future dates.
                                            X_scaled = scaler.transform(adjusted_features[feature_names])
                                            new_predictions = model.predict(X_scaled)
                                            adjusted_power = pd.Series(new_predictions, index=adjusted_features.index)
                                
                                    # Update progress bar
                                    progress_bar.progress((i + 1) / num_years, text=f"Simulating year {start_of_year.year}...")

                                return adjusted_power, well_drilling_dates, adjusted_features

                            # Prepare data for the simulation function
                            future_features = scenario_data[scenario_data.index > split_date][selected_features]
                            future_power = future_data[target_col]

                            adjusted_future_power, pulses, adjusted_future_features = apply_well_drilling_strategy(
                                future_features,
                                future_power,
                                model,
                                scaler,
                                selected_features,
                                threshold,
                                muw_flowrate,
                                steam_percentage,
                                brine_col,
                                steam_col,
                                feature_trends
                            )
                            progress_placeholder.progress(1.0, text="Simulation complete!")
                            
                            # Combine historical and adjusted future series
                            adjusted_series = scenario_data[target_col].copy()
                            adjusted_series.loc[scenario_data.index > split_date] = adjusted_future_power
                            
                            # Round results before storing
                            adjusted_series = adjusted_series.round(4)
                            adjusted_future_features = adjusted_future_features.round(4)

                            # Store all simulation results in session state
                            st.session_state.adjusted_series = adjusted_series
                            st.session_state.pulses = pulses
                            st.session_state.adjusted_future_features = adjusted_future_features

                            # Also prepare and store the CSV data for the simulation
                            split_date = scenario_data.index[-1] - pd.DateOffset(years=years)
                            selected_features = [col for col in ts_data.columns if col != target_col]
                            hist_features = scenario_data[scenario_data.index <= split_date][selected_features]
                            full_adjusted_features = pd.concat([hist_features, adjusted_future_features])
                            download_df_sim = full_adjusted_features.copy()
                            download_df_sim[f'{target_col} (no makeup well)'] = scenario_data[target_col]
                            download_df_sim[f'{target_col} (with makeup wells)'] = adjusted_series
                            st.session_state.csv_with_sim = download_df_sim.to_csv()

                        if 'adjusted_series' in st.session_state:
                            # Retrieve results from session state
                            adjusted_series = st.session_state.adjusted_series
                            pulses = st.session_state.pulses
                            adjusted_future_features = st.session_state.adjusted_future_features
                            future_features = scenario_data[scenario_data.index > split_date][selected_features]
                            
                            # Plot baseline vs adjusted predictions
                            fig_well = go.Figure()
                            
                            # Downsample data for plotting if it's too large
                            if len(scenario_data) > 2000:
                                plot_target = scenario_data[target_col].resample('D').mean()
                                plot_adjusted = adjusted_series.resample('D').mean()
                            else:
                                plot_target = scenario_data[target_col]
                                plot_adjusted = adjusted_series

                            fig_well.add_trace(
                                go.Scatter(
                                    x=plot_target.index,
                                    y=plot_target.values,
                                    name='Original Predictions',
                                    mode='lines'
                                )
                            )
                            fig_well.add_trace(
                                go.Scatter(
                                    x=plot_adjusted.index,
                                    y=plot_adjusted.values,
                                    name='Adjusted with New Wells',
                                    mode='lines'
                                )
                            )
                            # Add vertical lines for pulses
                            for pulse_time in pulses:
                                fig_well.add_vline(x=pulse_time, line_color="red")

                            fig_well.update_layout(
                                title='Power Predictions with New Wells',
                                xaxis_title='Date',
                                yaxis_title=target_col,
                                hovermode='x unified'
                            )
                            st.plotly_chart(fig_well, use_container_width=True)

                            # Plot yearly averages of the adjusted predictions
                            st.subheader("Yearly Power Predictions with New Wells")
                            
                            # Resample the adjusted series to yearly frequency
                            yearly_adjusted_avg = adjusted_series.resample('Y').mean()
                            
                            fig_yearly_well = go.Figure()
                            fig_yearly_well.add_trace(
                                go.Scatter(
                                    x=yearly_adjusted_avg.index,
                                    y=yearly_adjusted_avg.round(4).values,
                                    name='Yearly Average',
                                    mode='lines',
                                    line=dict(color='darkblue')
                                )
                            )

                            # Add vertical lines for pulses
                            for pulse_time in pulses:
                                fig_yearly_well.add_vline(x=pulse_time, line_color="red")
                            
                            # Add a line for the threshold
                            fig_yearly_well.add_hline(
                                y=threshold,
                                line_dash="dot",
                                line_color="red",
                                annotation_text=f"Drilling Threshold: {threshold:.2f} MW",
                                annotation_position="bottom right"
                            )
                            
                            fig_yearly_well.update_layout(
                                title='Yearly Average Power Predictions with New Wells',
                                xaxis_title='Year',
                                yaxis_title=f'Average {target_col}',
                                showlegend=False,
                                hovermode='x unified'
                            )
                            st.plotly_chart(fig_yearly_well, use_container_width=True)

                            # Plot quarterly averages of the adjusted predictions
                            st.subheader("Quarterly Power Predictions with New Wells")
                            quarterly_adjusted_avg = adjusted_series.resample('3M').mean()
                            fig_quarterly_well = go.Figure()
                            fig_quarterly_well.add_trace(
                                go.Scatter(
                                    x=quarterly_adjusted_avg.index,
                                    y=quarterly_adjusted_avg.round(4).values,
                                    name='Quarterly Average',
                                    mode='lines',
                                    line=dict(color='darkcyan')
                                )
                            )

                            for pulse_time in pulses:
                                fig_quarterly_well.add_vline(x=pulse_time, line_color="red")
                            
                            fig_quarterly_well.add_hline(
                                y=threshold,
                                line_dash="dot",
                                line_color="red",
                                annotation_text=f"Drilling Threshold: {threshold:.2f} MW",
                                annotation_position="bottom right"
                            )
                            
                            fig_quarterly_well.update_layout(
                                title='Quarterly Average Power Predictions with New Wells',
                                xaxis_title='Quarter',
                                yaxis_title=f'Average {target_col}',
                                showlegend=False,
                                hovermode='x unified'
                            )
                            st.plotly_chart(fig_quarterly_well, use_container_width=True)

                            # Plot adjusted input features
                            st.subheader("Adjusted Input Features from New Wells")

                            # Downsample data for plotting if it's too large
                            if len(future_features) > 2000:
                                plot_future_features = future_features.resample('D').mean()
                                plot_adjusted_features = adjusted_future_features.resample('D').mean()
                            else:
                                plot_future_features = future_features
                                plot_adjusted_features = adjusted_future_features

                            # Plot for Brine Flowrate
                            fig_brine = go.Figure()
                            fig_brine.add_trace(go.Scatter(x=plot_future_features.index, y=plot_future_features[brine_col], name='Original Brine Flowrate', mode='lines', line=dict(color='blue')))
                            fig_brine.add_trace(go.Scatter(x=plot_adjusted_features.index, y=plot_adjusted_features[brine_col], name='Adjusted Brine Flowrate', mode='lines', line=dict(color='orange')))
                            for pulse_time in pulses:
                                fig_brine.add_vline(x=pulse_time, line_color="red")
                            fig_brine.update_layout(title='Brine Flowrate with New Wells', xaxis_title='Date', yaxis_title='Flowrate (T/h)', hovermode='x unified')
                            st.plotly_chart(fig_brine, use_container_width=True)

                            # Plot for NCG+Steam Flowrate
                            fig_ncg = go.Figure()
                            fig_ncg.add_trace(go.Scatter(x=plot_future_features.index, y=plot_future_features[steam_col], name='Original NCG+Steam Flowrate', mode='lines', line=dict(color='green')))
                            fig_ncg.add_trace(go.Scatter(x=plot_adjusted_features.index, y=plot_adjusted_features[steam_col], name='Adjusted NCG+Steam Flowrate', mode='lines', line=dict(color='purple')))
                            for pulse_time in pulses:
                                fig_ncg.add_vline(x=pulse_time, line_color="red")
                            fig_ncg.update_layout(title='NCG+Steam Flowrate with New Wells', xaxis_title='Date', yaxis_title='Flowrate (T/h)', hovermode='x unified')
                            st.plotly_chart(fig_ncg, use_container_width=True)

                            st.subheader(f"Number of make-up wells to drill over {years} years: **{len(pulses)}**")

                            # Download button for data with well simulation
                            if 'csv_with_sim' in st.session_state:
                                st.download_button(
                                    label="Download Scenario Data (with MUW simulation)",
                                    data=st.session_state.csv_with_sim,
                                    file_name="scenario_predictions_with_MUW_simulation.csv",
                                    mime="text/csv",
                                    key="download_with_sim"
                                )

                        # ---------------------------------------------------------------

            except Exception as e:
                st.error(f"Error loading model data: {str(e)}")
                st.error("Please ensure the model training data exists and is compatible.")

if __name__ == "__main__":
    main()
