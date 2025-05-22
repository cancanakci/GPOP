import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from predict import load_model, predict
from data_prep import load_data, preprocess_data
import os
import json
from datetime import datetime
from plotly.subplots import make_subplots

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

    if model_option == "Use Default Model":
        st.write("Using the default pre-trained model for predictions.")
        # Load the default model
        model, scaler, feature_names = load_default_model("models")
        if model is not None and scaler is not None and feature_names is not None:
            st.sidebar.success("Default model loaded successfully.")
            
            # Load and display training data visualizations
            training_data = joblib.load(os.path.join("models", "default_training_data.pkl"))
            if training_data is not None:
                display_data_visualizations(training_data, model)
            
            # Show prediction options directly when using default model
            st.subheader("Make Predictions")
            input_method = st.radio(
                "Select input method:",
                ["Single Prediction", "Batch Prediction"]
            )

            if input_method == "Single Prediction":
                st.write("Enter values for prediction:")
                input_data = {}
                for feature in feature_names:
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        value=0.0,
                        format="%.4f"
                    )

                if st.button("Predict"):
                    try:
                        # Create a DataFrame from input data
                        input_df = pd.DataFrame([input_data])
                        
                        # Scale the features
                        scaled_input_features = scaler.transform(input_df)
                        scaled_input_df = pd.DataFrame(scaled_input_features, columns=feature_names)

                        # Make prediction
                        prediction_value = predict(model, scaled_input_df)
                        st.success(f"Predicted Power Output: {prediction_value[0]:.2f} MW")

                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
            else:  # Batch Prediction
                prediction_file = st.file_uploader("Upload data for batch prediction (CSV or Excel)", type=['csv', 'xlsx'])
                
                if prediction_file is not None:
                    try:
                        # Load the prediction data
                        pred_df = load_data(prediction_file)
                        st.success("File loaded successfully!")
                        
                        # Display preview
                        st.write("Data Preview:")
                        st.dataframe(pred_df.head())
                        
                        # Use a unique key for the predict button
                        if st.button("Make Predictions", key="batch_predict"):
                            try:
                                # Prepare data for prediction
                                input_df = pred_df[feature_names].copy()
                                
                                # Scale the features
                                scaled_input_features = scaler.transform(input_df)
                                scaled_input_df = pd.DataFrame(scaled_input_features, columns=feature_names)
                                
                                # Make predictions
                                predictions = predict(model, scaled_input_df)
                                
                                # Create results DataFrame
                                results_df = pred_df.copy()
                                results_df['Predicted Power Output (MW)'] = predictions
                                
                                # Display results
                                st.success("Predictions completed!")
                                st.write("Prediction Results:")
                                st.dataframe(results_df)
                                
                                # Add download button for results
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

            # Load and display metrics in sidebar (only for default model)
            metrics_path = os.path.join("models", "default_metrics.json")
            metrics = None
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            if metrics:
                st.sidebar.title("Model Information")
                st.sidebar.write(f"Model Type: {metrics['model_type']}")
                st.sidebar.write(f"Training Date: {datetime.strptime(metrics['timestamp'], '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Performance Metrics Section
                st.sidebar.subheader("Performance Metrics")
                
                # Test Set Metrics
                st.sidebar.write("Test Set Performance:")
                st.sidebar.write(f"R² Score: {metrics['metrics']['r2']:.4f}")
                st.sidebar.write(f"RMSE: {metrics['metrics']['rmse']:.4f}")
                st.sidebar.write(f"MSE: {metrics['metrics']['mse']:.4f}")
                
                # Cross-validation Metrics
                st.sidebar.write("Cross-validation Performance:")
                cv_metrics = metrics['metrics']['cv_metrics']
                st.sidebar.write(f"Mean R²: {cv_metrics['r2_mean']:.4f} (±{cv_metrics['r2_std'] * 2:.4f})")
                st.sidebar.write(f"Mean RMSE: {cv_metrics['rmse_mean']:.4f} (±{cv_metrics['rmse_std'] * 2:.4f})")
                
                # Individual CV Scores
                with st.sidebar.expander("Individual CV Scores"):
                    st.write("R² Scores per fold:")
                    for i, score in enumerate(cv_metrics['cv_scores']['r2'], 1):
                        st.write(f"Fold {i}: {score:.4f}")
                    st.write("RMSE Scores per fold:")
                    for i, score in enumerate(cv_metrics['cv_scores']['rmse'], 1):
                        st.write(f"Fold {i}: {score:.4f}")
                
                # Display model parameters in an expander
                with st.sidebar.expander("Model Parameters"):
                    for param, value in metrics['model_params'].items():
                        st.write(f"{param}: {value}")
                
                # Display actual vs predicted plot if available
                if 'actual' in metrics and 'predicted' in metrics:
                    actual = metrics['actual']
                    predicted = metrics['predicted']
                    df_plot = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
                    fig = px.scatter(df_plot, x='Actual', y='Predicted',
                                     title='Actual vs Predicted',
                                     labels={'Actual': 'Actual Brüt Güç', 'Predicted': 'Predicted Brüt Güç'})
                    fig.add_shape(type='line', x0=min(actual), y0=min(actual), x1=max(actual), y1=max(actual),
                                  line=dict(color='black', dash='dash'))
                    st.sidebar.plotly_chart(fig, use_container_width=True)
                
                # Display loss curve if available
                if 'loss_curve' in metrics['metrics']:
                    loss_curve = metrics['metrics']['loss_curve']
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
    else:  # Train New Model
        st.write("Upload your data to train a new model.")
        training_file = st.file_uploader("Upload training data (CSV or Excel)", type=['csv', 'xlsx'], key="training_file")
        
        if training_file is not None:
            try:
                # Load and preprocess the data directly from the uploaded file
                df = load_data(training_file)
                st.success("File loaded successfully!")
                
                # Display preview
                st.write("Data Preview:")
                st.dataframe(df.head())
                
                # Display dataset information
                st.write("Dataset Information:")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Number of rows: {len(df)}")
                    st.write(f"Number of features: {len(df.columns)}")
                with col2:
                    st.write("Feature names:")
                    for col in df.columns:
                        st.write(f"- {col}")
                
                # Display basic statistics
                st.write("Basic Statistics:")
                st.dataframe(df.describe())
                
                # Create training data dictionary for visualization
                target_column = df.columns[-1]  # Get the last column name
                training_data = {
                    'X_train': df.drop(target_column, axis=1),
                    'X_test': df.drop(target_column, axis=1).iloc[:len(df)//5],  # Use 20% for test
                    'y_train': df[target_column],
                    'y_test': df[target_column].iloc[:len(df)//5],
                    'feature_names': df.drop(target_column, axis=1).columns.tolist(),
                    'target_column': target_column
                }
                
                # Use session state to persist new model info
                if 'new_model_metrics' not in st.session_state:
                    st.session_state['new_model_metrics'] = None
                if 'new_model_trained' not in st.session_state:
                    st.session_state['new_model_trained'] = False
                if st.button("Train New Model"):
                    try:
                        from train import train_model
                        metrics = train_model(training_file, "models")
                        model, scaler, feature_names = load_latest_model_files("models")
                        st.success("New model trained and loaded successfully!")
                        cleanup_old_models("models")
                        st.session_state['new_model_metrics'] = metrics
                        st.session_state['new_model_trained'] = True
                        st.session_state['new_model_training_data'] = training_data
                        st.session_state['new_model_scaler'] = scaler
                        st.session_state['new_model_model'] = model
                        st.session_state['new_model_feature_names'] = feature_names
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
                # If a new model has been trained, show its info and prediction UI
                if st.session_state.get('new_model_trained', False):
                    metrics = st.session_state['new_model_metrics']
                    training_data = st.session_state['new_model_training_data']
                    scaler = st.session_state['new_model_scaler']
                    model = st.session_state['new_model_model']
                    feature_names = st.session_state['new_model_feature_names']
                    st.subheader("Training Results")
                    st.write(f"R² Score: {metrics['metrics']['r2']:.4f}")
                    st.write(f"RMSE: {metrics['metrics']['rmse']:.4f}")
                    display_data_visualizations(training_data, model)
                    st.sidebar.title("Model Information (New Model)")
                    st.sidebar.write(f"Model Type: {metrics['model_type']}")
                    st.sidebar.write(f"Training Date: {datetime.strptime(metrics['timestamp'], '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')}")
                    st.sidebar.subheader("Performance Metrics")
                    st.sidebar.write("Test Set Performance:")
                    st.sidebar.write(f"R² Score: {metrics['metrics']['r2']:.4f}")
                    st.sidebar.write(f"RMSE: {metrics['metrics']['rmse']:.4f}")
                    st.sidebar.write(f"MSE: {metrics['metrics']['mse']:.4f}")
                    st.sidebar.write("Cross-validation Performance:")
                    cv_metrics = metrics['metrics']['cv_metrics']
                    st.sidebar.write(f"Mean R²: {cv_metrics['r2_mean']:.4f} (±{cv_metrics['r2_std'] * 2:.4f})")
                    st.sidebar.write(f"Mean RMSE: {cv_metrics['rmse_mean']:.4f} (±{cv_metrics['rmse_std'] * 2:.4f})")
                    with st.sidebar.expander("Individual CV Scores"):
                        st.write("R² Scores per fold:")
                        for i, score in enumerate(cv_metrics['cv_scores']['r2'], 1):
                            st.write(f"Fold {i}: {score:.4f}")
                        st.write("RMSE Scores per fold:")
                        for i, score in enumerate(cv_metrics['cv_scores']['rmse'], 1):
                            st.write(f"Fold {i}: {score:.4f}")
                    with st.sidebar.expander("Model Parameters"):
                        for param, value in metrics['model_params'].items():
                            st.write(f"{param}: {value}")
                    if 'actual' in metrics and 'predicted' in metrics:
                        actual = metrics['actual']
                        predicted = metrics['predicted']
                        df_plot = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
                        fig = px.scatter(df_plot, x='Actual', y='Predicted',
                                         title='Actual vs Predicted',
                                         labels={'Actual': 'Actual Brüt Güç', 'Predicted': 'Predicted Brüt Güç'})
                        fig.add_shape(type='line', x0=min(actual), y0=min(actual), x1=max(actual), y1=max(actual),
                                      line=dict(color='black', dash='dash'))
                        st.sidebar.plotly_chart(fig, use_container_width=True)
                    
                    # Display loss curve if available
                    if 'loss_curve' in metrics['metrics']:
                        loss_curve = metrics['metrics']['loss_curve']
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
                    st.subheader("Make Predictions")
                    input_method = st.radio(
                        "Select input method:",
                        ["Single Prediction", "Batch Prediction"],
                        key="new_model_predict_method"
                    )
                    if input_method == "Single Prediction":
                        st.write("Enter values for prediction:")
                        input_data = {}
                        for feature in feature_names:
                            input_data[feature] = st.number_input(
                                f"{feature}",
                                value=0.0,
                                format="%.4f",
                                key=f"new_model_input_{feature}"
                            )
                        if st.button("Predict", key="new_model_single_predict"):
                            try:
                                input_df = pd.DataFrame([input_data])
                                scaled_input_features = scaler.transform(input_df)
                                scaled_input_df = pd.DataFrame(scaled_input_features, columns=feature_names)
                                prediction_value = predict(model, scaled_input_df)
                                st.success(f"Predicted Power Output: {prediction_value[0]:.2f} MW")
                            except Exception as e:
                                st.error(f"Error making prediction: {str(e)}")
                    else:  # Batch Prediction
                        prediction_file = st.file_uploader("Upload data for batch prediction (CSV or Excel)", type=['csv', 'xlsx'], key="new_model_batch_file")
                        if prediction_file is not None:
                            try:
                                pred_df = load_data(prediction_file)
                                st.success("File loaded successfully!")
                                st.write("Data Preview:")
                                st.dataframe(pred_df.head())
                                if st.button("Make Predictions", key="new_model_batch_predict"):
                                    try:
                                        input_df = pred_df[feature_names].copy()
                                        scaled_input_features = scaler.transform(input_df)
                                        scaled_input_df = pd.DataFrame(scaled_input_features, columns=feature_names)
                                        predictions = predict(model, scaled_input_df)
                                        results_df = pred_df.copy()
                                        results_df['Predicted Power Output (MW)'] = predictions
                                        st.success("Predictions completed!")
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
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        else:
            st.info("Please upload a training data file to train a new model.")

if __name__ == "__main__":
    main()
