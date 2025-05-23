import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from predict import load_model, predict, check_input_values
from data_prep import load_data, preprocess_data
import os
import json
from datetime import datetime
from plotly.subplots import make_subplots

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
    
    # Correlation Heatmap
    corr = df_combined.corr()
    fig = px.imshow(corr, text_auto=True, title='Feature Correlation Heatmap')
    fig.update_layout(
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_nticks=len(corr.columns),
        yaxis_nticks=len(corr.index)
    )
    st.plotly_chart(fig)
    
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

def display_prediction_visualizations(results_df, target_column='Target'):
    """Display visualizations for batch prediction results."""
    st.subheader("Prediction Results Visualizations")
    
    # Distribution of predictions
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=('Prediction Distribution', 'Prediction Box Plot'))
    
    # Add histogram
    fig.add_trace(
        go.Histogram(x=results_df['Predicted Power Output (MW)'], name='Histogram'),
        row=1, col=1
    )
    
    # Add box plot
    fig.add_trace(
        go.Box(y=results_df['Predicted Power Output (MW)'], name='Box Plot'),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text='Predicted Power Output Analysis'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Time series plot if there's a time column
    if 'Tarih' in results_df.columns:
        fig = px.line(results_df, x='Tarih', y='Predicted Power Output (MW)',
                     title='Predicted Power Output Over Time')
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation with input features
    st.write("Feature Correlations with Predictions")
    corr = results_df.corr()['Predicted Power Output (MW)'].sort_values(ascending=False)
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

    if input_method == "Single Prediction":
        st.write("Enter values for prediction:")
        input_data = {}

        if training_data is not None:
            X_train = training_data['X_train']

            for feature in feature_names:
                train_q1 = X_train[feature].quantile(0.25)
                train_q3 = X_train[feature].quantile(0.75)
                train_iqr = train_q3 - train_q1
                iqr_lower = train_q1 - 1.5 * train_iqr
                iqr_upper = train_q3 + 1.5 * train_iqr

                input_data[feature] = st.number_input(
                    f"{feature} (Typical Range: {iqr_lower:.2f} - {iqr_upper:.2f})",
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

                st.success(f"Predicted Power Output: {prediction_value[0]:.2f} MW")

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
                        input_df = pred_df[feature_names].copy()

                        warning_flags_df, yellow_warnings, red_warnings, warning_ranges = check_input_values(input_df, training_data)

                        scaled_input_features = scaler.transform(input_df)
                        scaled_input_df = pd.DataFrame(scaled_input_features, columns=feature_names)
                        predictions = predict(model, scaled_input_df)
                        results_df = pred_df.copy()
                        results_df['Predicted Power Output (MW)'] = predictions

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

def main():
    # Set page configuration
    st.set_page_config(
        page_title="GPOP",
        page_icon="💨",
        layout="wide"
    )

    st.title("Geothermal Power Output Prediction")

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
                df = load_data(training_file)
                st.success("File loaded successfully!")

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

                target_column = df.columns[-1]
                training_data_for_viz = {
                    'X_train': df.drop(target_column, axis=1),
                    'X_test': df.drop(target_column, axis=1).iloc[:len(df)//5],
                    'y_train': df[target_column],
                    'y_test': df[target_column].iloc[:len(df)//5],
                    'feature_names': df.drop(target_column, axis=1).columns.tolist(),
                    'target_column': target_column
                }

                if 'new_model_metrics' not in st.session_state:
                    st.session_state['new_model_metrics'] = None
                if 'new_model_trained' not in st.session_state:
                    st.session_state['new_model_trained'] = False

                if st.button("Train New Model"):
                    try:
                        from train import train_model
                        metrics = train_model(training_file, models_dir)
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
                st.error(f"Error loading file: {str(e)}")
        else:
            st.info("Please upload a training data file to train a new model.")

    # Load the selected model components
    model, scaler, feature_names, training_data, status = load_selected_model_components(model_option, models_dir)

    st.sidebar.write(status)

    if model and scaler and feature_names and training_data:
        # Display training data visualizations
        display_data_visualizations(training_data, model)

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

if __name__ == "__main__":
    main()
