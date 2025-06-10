import os
import joblib
import streamlit as st
import json
from tensorflow.keras.models import load_model as keras_load_model

def load_model(model_path):
    """Load a model from a file, supporting both .pkl and .keras formats."""
    if model_path.endswith('.pkl'):
        return joblib.load(model_path)
    elif model_path.endswith('.keras'):
        return keras_load_model(model_path)
    else:
        raise ValueError(f"Unsupported model file format for {model_path}")

def load_default_model(models_dir):
    """Load the default model files."""
    try:
        metrics_path = os.path.join(models_dir, "default_metrics.json")
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        model_path = metrics.get('model_path', os.path.join(models_dir, "default_model.pkl")) # Fallback for old metrics
        model = load_model(model_path)
        scaler = joblib.load(os.path.join(models_dir, "default_scaler.pkl"))
        feature_names = joblib.load(os.path.join(models_dir, "default_feature_names.pkl"))
        return model, scaler, feature_names
    except Exception as e:
        print(f"Error loading default model files via metrics: {e}")
        try:
            # Fallback for old default model format without metrics.json
            model = joblib.load(os.path.join(models_dir, "default_model.pkl"))
            scaler = joblib.load(os.path.join(models_dir, "default_scaler.pkl"))
            feature_names = joblib.load(os.path.join(models_dir, "default_feature_names.pkl"))
            print("Successfully loaded default model using fallback method.")
            return model, scaler, feature_names
        except Exception as e_fallback:
            print(f"Fallback loading for default model failed: {e_fallback}")
            return None, None, None

def load_latest_model_files(models_dir):
    """Load the latest model, scaler, and feature names files from the models directory."""
    metrics_files = [f for f in os.listdir(models_dir) if f.startswith('metrics_') and f.endswith('.json')]
    if not metrics_files:
        return None, None, None
    
    # Sort by creation time to get the latest
    latest_metrics_file = sorted(metrics_files, key=lambda f: os.path.getmtime(os.path.join(models_dir, f)))[-1]
    
    try:
        with open(os.path.join(models_dir, latest_metrics_file), 'r') as f:
            metrics = json.load(f)
        
        model = load_model(metrics['model_path'])
        
        timestamp = metrics['timestamp']
        scaler_path = os.path.join(models_dir, f"scaler_{timestamp}.pkl")
        feature_names_path = os.path.join(models_dir, f"feature_names_{timestamp}.pkl")

        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(feature_names_path)
        
        return model, scaler, feature_names
    except Exception as e:
        print(f"Error loading latest model files: {e}")
        return None, None, None

def cleanup_old_models(models_dir):
    """Clean up old model files, keeping only the default model and the latest trained model."""
    files_to_keep = {
        "default_model.pkl", "default_model.keras",
        "default_scaler.pkl", "default_feature_names.pkl",
        "default_metrics.json", "default_training_data.pkl"
    }
    
    all_files = os.listdir(models_dir)
    
    # Find the latest model by looking at the latest metrics file
    metrics_files = sorted([f for f in all_files if f.startswith('metrics_') and f.endswith('.json')])
    if metrics_files:
        latest_metrics_file = metrics_files[-1]
        files_to_keep.add(latest_metrics_file)
        
        try:
            with open(os.path.join(models_dir, latest_metrics_file), 'r') as f:
                metrics = json.load(f)

            files_to_keep.add(os.path.basename(metrics['model_path']))
            files_to_keep.add(os.path.basename(metrics['training_data_path']))
            
            timestamp = metrics['timestamp']
            files_to_keep.add(f"scaler_{timestamp}.pkl")
            files_to_keep.add(f"feature_names_{timestamp}.pkl")
        except Exception as e:
            print(f"Could not read latest metrics file for cleanup: {e}")

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
        # First, try to load the latest trained model from disk if the app was restarted
        model, scaler, feature_names = load_latest_model_files(models_dir)
        if model and scaler and feature_names:
             try:
                # find latest training data
                training_data_files = sorted([f for f in os.listdir(models_dir) if f.startswith('training_data_') and f.endswith('.pkl')])
                if training_data_files:
                    latest_training_data = training_data_files[-1]
                    training_data = joblib.load(os.path.join(models_dir, latest_training_data))
                    status = "Loaded latest trained model from disk."
                    # Put the loaded model into session state
                    st.session_state['new_model_trained'] = True
                    st.session_state['new_model_model'] = model
                    st.session_state['new_model_scaler'] = scaler
                    st.session_state['new_model_feature_names'] = feature_names
                    st.session_state['new_model_training_data'] = training_data
                else:
                    status = "Latest model loaded, but no training data found."

             except Exception as e:
                status = f"Error loading training data for latest model: {e}"
        
        # Then, check session state for a model trained in the current session
        if st.session_state.get('new_model_trained', False):
            model = st.session_state.get('new_model_model')
            scaler = st.session_state.get('new_model_scaler')
            feature_names = st.session_state.get('new_model_feature_names')
            training_data = st.session_state.get('new_model_training_data')
            if model and scaler and feature_names and training_data:
                status = "Newly trained model is active."
            else:
                status = "No new model has been trained yet."
        elif not model: # If no model was loaded from disk
            status = "No new model has been trained yet."

    return model, scaler, feature_names, training_data, status 