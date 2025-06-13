import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load data from various file formats with robust error handling."""
    try:
        # Handle file uploads (Streamlit)
        if hasattr(file_path, 'getvalue'):
            # Get the file extension and content type
            file_name = file_path.name.lower()
            print(f"Attempting to load file: {file_name}")  # Debug print
            
            # Check if it's an Excel file by looking at the first few bytes
            file_path.seek(0)
            header = file_path.read(8)
            file_path.seek(0)  # Reset file pointer
            
            # Excel files start with PK\x03\x04 (ZIP file signature)
            is_excel = header.startswith(b'PK\x03\x04')
            
            if is_excel or file_name.endswith(('.xlsx', '.xls')):
                try:
                    # Try reading with openpyxl first
                    df = pd.read_excel(file_path, engine='openpyxl')
                    print(f"Successfully read Excel file with openpyxl")  # Debug print
                    print(f"Columns found: {df.columns.tolist()}")  # Debug print
                    if len(df.columns) > 1:
                        return df
                except Exception as e:
                    print(f"Failed to read with openpyxl: {str(e)}")  # Debug print
                    try:
                        # Reset file pointer
                        file_path.seek(0)
                        # Try reading with xlrd as fallback
                        df = pd.read_excel(file_path, engine='xlrd')
                        print(f"Successfully read Excel file with xlrd")  # Debug print
                        print(f"Columns found: {df.columns.tolist()}")  # Debug print
                        if len(df.columns) > 1:
                            return df
                    except Exception as e:
                        print(f"Failed to read with xlrd: {str(e)}")  # Debug print
                        raise ValueError(f"Could not read Excel file with any engine: {str(e)}")
            
            elif file_name.endswith('.csv'):
                # First try to read the raw content to check the format
                content = file_path.getvalue().decode('utf-8', errors='ignore')
                print(f"First 100 characters of file: {content[:100]}")  # Debug print
                
                # Try to detect the delimiter
                first_line = content.split('\n')[0]
                print(f"First line: {first_line}")  # Debug print
                
                # Try different delimiters
                delimiters = [';', ',', '\t', '|']  # Prioritize semicolon
                for delimiter in delimiters:
                    try:
                        # Reset file pointer
                        file_path.seek(0)
                        # Add decimal=',' to handle comma decimal separator
                        df = pd.read_csv(file_path, sep=delimiter, encoding='utf-8', decimal=',')
                        print(f"Successfully read with delimiter: {delimiter}")  # Debug print
                        print(f"Columns found: {df.columns.tolist()}")  # Debug print
                        if len(df.columns) > 1:
                            return df
                    except Exception as e:
                        print(f"Failed with delimiter {delimiter}: {str(e)}")  # Debug print
                        continue
                
                # If all delimiters fail, try with python engine
                try:
                    file_path.seek(0)
                    # Add decimal=',' to handle comma decimal separator
                    df = pd.read_csv(file_path, sep=None, engine='python', decimal=',')
                    print("Successfully read with python engine")  # Debug print
                    print(f"Columns found: {df.columns.tolist()}")  # Debug print
                    if len(df.columns) > 1:
                        return df
                except Exception as e:
                    print(f"Failed with python engine: {str(e)}")  # Debug print
                
                raise ValueError("Could not read CSV file with any supported delimiter")
            
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files.")
        
        # Handle file paths
        elif isinstance(file_path, str):
            if file_path.endswith('.csv'):
                # Try different delimiters for CSV
                delimiters = [';', ',', '\t', '|']  # Prioritize semicolon
                for delimiter in delimiters:
                    try:
                        # Add decimal=',' to handle comma decimal separator
                        df = pd.read_csv(file_path, sep=delimiter, encoding='utf-8', decimal=',')
                        if len(df.columns) > 1:
                            return df
                    except:
                        continue
                raise ValueError("Could not read CSV file with any supported delimiter")
            
            elif file_path.endswith(('.xlsx', '.xls')):
                try:
                    # Try openpyxl first
                    # Add decimal=',' to handle comma decimal separator
                    df = pd.read_excel(file_path, engine='openpyxl', decimal=',')
                    if len(df.columns) > 1:
                        return df
                except Exception as e:
                    try:
                        # Try xlrd as fallback
                        # Add decimal=',' to handle comma decimal separator
                        df = pd.read_excel(file_path, engine='xlrd', decimal=',')
                        if len(df.columns) > 1:
                            return df
                    except Exception as e:
                        raise ValueError(f"Could not read Excel file with any engine: {str(e)}")
            
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files.")
        
        raise ValueError("Invalid file input")
    
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        raise

    # Force numeric columns to be parsed as numeric - Keep this as a fallback
    for col in df.columns:
        # Only attempt conversion if the dtype is not already numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # After trying to convert, check if any numeric columns exist
    numeric_cols_after_conversion = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_cols_after_conversion:
        raise ValueError("No numeric columns found even after attempting conversion.")
        
    return df

def preprocess_data(df):
    """Perform preprocessing on the dataframe. Tries interpolation first, then mean imputation."""
    processed_df = df.copy()

    # Debug: Print DataFrame dtypes
    print("DataFrame dtypes:")
    print(processed_df.dtypes)

    # Drop non-numeric columns (e.g., datetime columns)
    numeric_df = processed_df.select_dtypes(include=['number'])
    print("Numeric columns found:", numeric_df.columns.tolist())
    if numeric_df.empty:
        raise ValueError("No numeric features available after dropping non-numeric columns.")
    processed_df = numeric_df

    # Handle missing values using linear interpolation
    processed_df.interpolate(inplace=True)

    # Handle any remaining NaNs with mean imputation
    for col in processed_df.columns:
         processed_df[col].fillna(processed_df[col].mean(), inplace=True)

    return processed_df
