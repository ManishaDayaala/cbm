#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import shutil
import pandas as pd
from datetime import datetime
import streamlit as st

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib


# Set a random seed for reproducibility
def set_random_seed(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)

# Define the main folder path
MAINFOLDER = r"./APPdata"

# Create other paths relative to the main folder
training_file_path = os.path.join(MAINFOLDER, "Training", "Training.xlsx")  # FIXED TRAINING DATA
test_file_path = os.path.join(MAINFOLDER, "24hrData", "Dailydata.xlsx")  # DAILY DATA
excel_file_path = os.path.join(MAINFOLDER, "Breakdownrecords.xlsx")  # Recording excel for BD
folderpath = os.path.join(MAINFOLDER, "TemporaryData")  # Temporary dump files collector
model_folder_path = os.path.join(MAINFOLDER, "Models")
uploaded_files = []  # List to keep track of uploaded files

# Streamlit UI
st.title("File Upload and Preprocessing")
st.markdown("Upload your files, and they will be preprocessed accordingly.")

# File Upload Section
uploaded_files = st.file_uploader("Upload Excel files", type=['xlsx'], accept_multiple_files=True)

# Show status
status_placeholder = st.empty()

# Function to clear old files from the folder
def clear_saved_files():
    try:
        # Clear old files in the folder
        for filename in os.listdir(folderpath):
            file_path = os.path.join(folderpath, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)  # Remove the file
            except Exception as e:
                status_placeholder.error(f"Error clearing old files: {e}")
                return
        status_placeholder.success("Saved files cleared successfully!")
    except Exception as e:
        status_placeholder.error(f"Error: {e}")

# Function to handle file saving (clear old files before saving new ones)
def save_files(uploaded_files):
    try:
        if not uploaded_files:
            status_placeholder.error("No files to save!")
            return

        # Clear old files in the folder before saving new files
        clear_saved_files()

        # Save each file from the uploaded list to the target folder
        for file in uploaded_files:
            with open(os.path.join(folderpath, file.name), "wb") as f:
                f.write(file.getbuffer())

        status_placeholder.success("Files saved successfully!")

    except Exception as e:
        status_placeholder.error(f"Error: {e}")

# Function to clear uploaded files list
def clear_uploaded_files():
    global uploaded_files
    uploaded_files = []  # Clear the list
    status_placeholder.success("Uploaded files list cleared!")

# Function to remove the last uploaded file from the list
def clear_last_uploaded_file():
    global uploaded_files
    if uploaded_files:
        uploaded_files.pop()  # Remove the last file from the list
        status_placeholder.success("Last uploaded file removed!")
    else:
        status_placeholder.warning("No files to remove!")

## Streamlit buttons and actions
#if st.button("Clear Uploaded Files"):
#    clear_uploaded_files()
#
#if st.button("Clear Last Uploaded File"):
#    clear_last_uploaded_file()

if st.button("Save Files"):
    if uploaded_files:
        save_files(uploaded_files)
    else:
        st.error("Please upload files first.")


################# Data Preprocessing ##########################

# Process files and apply preprocessing logic

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

def process_data():
    
    # Define the input file (only one file in the folder)
    input_file_name = os.listdir(folderpath)[0]  # Assuming only one file in the folder
    input_file_path = os.path.join(folderpath, input_file_name)

    # Check if the input file exists
    if not os.path.isfile(input_file_path):
        st.error(f"Input file '{input_file_name}' does not exist!")
        return

    # List of 12 unique asset names
    assets_list = [
        "A1 GM 1 GB IP DE", "A1 GM1 MDE", "A1 GM 2 GB IP DE", "A1 GM2 MDE",
        "A1 GM 3 GB IP DE", "A1 GM3 MDE", "A1 GM 4 GB IP DE", "A1 GM4 MDE",
        "A1 GM 5 GB IP DE", "A1 GM5 MDE", "A1 GM 6 GB IP DE", "A1 GM6 MDE"
    ]

    # Columns to extract for each asset, corresponding to F, I, L, O, R, U
    required_column_indices = [5, 8, 11, 14, 17, 20]  # 0-based indices for F, I, L, O, R, U
    required_column_names = ['a2', 'vv2', 'av2', 'hv2', 't2', 'd2']

    # Check if the output folder exists, if not, create it
    #if not os.path.isdir(test_file_path):
        #os.makedirs(test_file_path)
        #st.info(f"Output folder '{test_file_path}' created!")

    # Load the input file
    input_df = pd.read_excel(input_file_path)

    # Initialize an empty DataFrame to store combined data
    output_df = pd.DataFrame()

    # Loop over each asset in assets_list
    for asset_name in assets_list:
        # Find rows where Column B (index 1) matches the asset_name
        asset_rows = input_df[input_df.iloc[:, 1] == asset_name]
        
        # Check if any rows were found
        if not asset_rows.empty:
            # Parse the date and time from Column C (index 2)
            asset_rows['DateTime'] = pd.to_datetime(asset_rows.iloc[:, 2], format='%d-%m-%Y %H:%M')
            
            # Identify the earliest start time in the data for this asset
            start_time = asset_rows['DateTime'].min().replace(hour=5, minute=30)
            end_time = start_time + timedelta(days=1, hours=0, minutes=0)
            
            # Filter rows within this 24-hour window (from earliest 5:30 AM to the next day 5:30 AM)
            filtered_rows = asset_rows[(asset_rows['DateTime'] >= start_time) & (asset_rows['DateTime'] <= end_time)]
            
            # Select only the first 49 rows if there are more than 49 available
            filtered_rows = filtered_rows.head(49)
            
            # Collect only the specified columns (F, I, L, O, R, U) for the 49 rows
            data_for_asset = filtered_rows.iloc[:, required_column_indices].values
            data_for_asset = pd.DataFrame(data_for_asset, columns=required_column_names)
            
            # Fill any missing rows with 0s if there are fewer than 49 rows
            if len(data_for_asset) < 49:
                missing_rows = 49 - len(data_for_asset)
                data_for_asset = pd.concat([data_for_asset, pd.DataFrame(0, index=range(missing_rows), columns=required_column_names)], ignore_index=True)
        else:
            # If no rows found for this asset, fill with 0s for all columns
            data_for_asset = pd.DataFrame(0, index=range(49), columns=required_column_names)

        # Rename columns to reflect asset-specific names (e.g., "a2" becomes "A1 GM 1 GB IP DE_a2")
        data_for_asset.columns = [f"{asset_name}_{col}" for col in required_column_names]

        # Concatenate the data for this asset horizontally to the output DataFrame
        output_df = pd.concat([output_df, data_for_asset], axis=1)

    # Generate Date, Time, and Sr No columns at 30-minute intervals
    date_list = [(start_time + timedelta(minutes=30 * i)).strftime('%d-%m-%Y') for i in range(49)]
    time_list = [(start_time + timedelta(minutes=30 * i)).strftime('%I:%M %p') for i in range(49)]
    sr_no_list = list(range(1, 50))


    # Insert Date, Time, and Sr No columns into the final output DataFrame
    output_df.insert(0, 'Date', date_list)
    output_df.insert(1, 'Time', time_list)
    output_df.insert(2, 'Sr No', sr_no_list)

    # Add an empty 'Code' column at the end
    output_df['Code'] = ''


    # Save the processed data using ExcelWriter
    with pd.ExcelWriter(test_file_path, engine='openpyxl') as writer:
        output_df.to_excel(writer, index=False)

    # Save the processed file to the output folder
    #output_file_path = os.path.join(test_file_path, f"processed_{input_file_name}")
    #output_df.to_excel(output_file_path, index=False)
    #st.success(f"Processed file saved as '{output_file_path}'")

    # Display success message when all files are processed
    st.info(f"Data has been processed and saved to {test_file_path}")

# Streamlit UI to trigger the data preprocessing
st.title('Asset Data Processing')
st.write("Click the button below to preprocess the data from the input file.")

# Create a button to trigger the process
if st.button('Preprocess Data'):
    process_data()




################breakdown records###########################


import streamlit as st
import pandas as pd
from datetime import datetime

# Path to the Excel file
#excel_file_path = "breakdown_data.xlsx"

# Function to save breakdown data to Excel
def save_breakdown_data():
    date = st.session_state.date_entry.strftime("%d-%m-%y")
    time = f"{st.session_state.hour_combobox}:{st.session_state.minute_combobox} {st.session_state.am_pm_combobox}"
    code = st.session_state.code_entry
    
    if not code:
        st.session_state.status = "Please fill the Breakdown Code!"
        st.session_state.status_color = "red"
        return

    try:
        df = pd.read_excel(excel_file_path)
        new_row = pd.DataFrame([[date, time, code]], columns=["Date", "Time", "Code"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(excel_file_path, index=False)
        st.session_state.status = "Breakdown data saved successfully!"
        st.session_state.status_color = "green"
    except Exception as e:
        st.session_state.status = f"Error: {e}"
        st.session_state.status_color = "red"

# Function to clear the breakdown input fields
def clear_breakdown_fields():
    st.session_state.date_entry = datetime.now()
    st.session_state.hour_combobox = '12'
    st.session_state.minute_combobox = '00'
    st.session_state.am_pm_combobox = 'AM'
    st.session_state.code_entry = ''
    st.session_state.status = "Fields cleared!"
    st.session_state.status_color = "blue"

# Streamlit UI Setup
def display_ui():
    # Initialize session state if not already initialized
    if 'status' not in st.session_state:
        st.session_state.status = ""
        st.session_state.status_color = "black"
        st.session_state.date_entry = datetime.now()
        st.session_state.hour_combobox = '12'
        st.session_state.minute_combobox = '00'
        st.session_state.am_pm_combobox = 'AM'
        st.session_state.code_entry = ''

    st.title("Breakdown Record")

    # Date input
    st.session_state.date_entry = st.date_input("Date", value=st.session_state.date_entry)
    
    # Time selection
    time_column1, time_column2, time_column3 = st.columns(3)
    with time_column1:
        st.session_state.hour_combobox = st.selectbox("Hour", options=[f"{i:02d}" for i in range(1, 13)], index=int(st.session_state.hour_combobox)-1)
    with time_column2:
        st.session_state.minute_combobox = st.selectbox("Minute", options=[f"{i:02d}" for i in range(0, 60, 5)], index=int(st.session_state.minute_combobox)//5)
    with time_column3:
        st.session_state.am_pm_combobox = st.selectbox("AM/PM", options=["AM", "PM"], index=["AM", "PM"].index(st.session_state.am_pm_combobox))

    # Breakdown code input
    st.session_state.code_entry = st.text_input("Breakdown Code", value=st.session_state.code_entry)

    # Status display (Feedback to user)
    st.markdown(f"<p style='color:{st.session_state.status_color};'>{st.session_state.status}</p>", unsafe_allow_html=True)

    # Buttons for saving and clearing
    col1, col2 = st.columns(2)
    with col1:
        save_button = st.button("Save Breakdown")
        if save_button:
            save_breakdown_data()
    with col2:
        clear_button = st.button("Clear Fields")
        if clear_button:
            clear_breakdown_fields()

# Run the UI display
if __name__ == "__main__":
    display_ui()



################################        time prediction             #############################




import streamlit as st
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from datetime import datetime
import numpy as np

# Define the path to save models within the main folder


# Predefined file paths (change these paths to the actual file locations on your system)
#training_file_path = "path_to_your_training_file.xlsx"  # Replace with actual path
#test_file_path = "path_to_your_test_file.xlsx"  # Replace with actual path

# Function to set random seed for reproducibility
def set_random_seed(seed=42):
    np.random.seed(seed)

# Define the training function
def train_model(training_file_path):
    def load_data(file_path):
        df = pd.read_excel(file_path, sheet_name="Time")
        X = df.iloc[:, 1:72].values
        y = df.iloc[:, 73].values
        return X, y

    def preprocess_data(X, y):
        mask = y < 90  # Time to breakdown less than 72 hours
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        # Use a fixed random_state to ensure reproducibility
        X_train, X_val, y_train, y_val = train_test_split(X_filtered, y_filtered, test_size=0.01, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        joblib.dump(scaler, os.path.join(model_folder_path, 'scalerfinT.pkl'))
        return X_train_scaled, X_val_scaled, y_train, y_val

    def build_model(input_shape):
        model = Sequential()
        model.add(Dense(128, input_dim=input_shape, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    # Set random seed for reproducibility
    set_random_seed()

    X, y = load_data(training_file_path)
    X_train, X_val, y_train, y_val = preprocess_data(X, y)
    model = build_model(X_train.shape[1])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping])
    model.save(os.path.join(model_folder_path, 'trained_modelFINT.h5'))

# Define the prediction function
def predict_time(test_file_path):
    def load_test_data(file_path):
        df = pd.read_excel(file_path)
        serial_numbers = df.iloc[:, 2].values
        times = df.iloc[:, 1].values
        X_test = df.iloc[:, 3:74].values
        return df, X_test, serial_numbers, times

    def preprocess_test_data(X_test):
        scaler = joblib.load(os.path.join(model_folder_path, 'scalerfinT.pkl'))
        X_test_scaled = scaler.transform(X_test)
        return X_test_scaled

    def predict_time_to_breakdown(X_test_scaled):
        model = load_model(os.path.join(model_folder_path, 'trained_modelFINT.h5'))
        predictions = model.predict(X_test_scaled)
        return predictions

    def calculate_time_difference(times, predictions):
        time_to_breakdown_with_time = []
        for time_str, prediction in zip(times, predictions):
            time_obj = datetime.strptime(time_str, '%I:%M %p')
            midnight = datetime.combine(time_obj.date(), datetime.min.time())
            time_difference = (time_obj - midnight).total_seconds() / 3600
            adjusted_time_to_bd = prediction[0] + time_difference
            time_to_breakdown_with_time.append(adjusted_time_to_bd)
        return time_to_breakdown_with_time

    def find_minimum_time(time_to_breakdown_with_time):
        min_time = min(time_to_breakdown_with_time)
        return min_time

    # Set random seed for reproducibility
    set_random_seed()

    try:
        # Load and preprocess the test data
        df, X_test, serial_numbers, times = load_test_data(test_file_path)
        X_test_scaled = preprocess_test_data(X_test)

        # Make predictions
        predictions = predict_time_to_breakdown(X_test_scaled)
        predictions_with_time = calculate_time_difference(times, predictions)

        # Find the minimum predicted time
        min_time = find_minimum_time(predictions_with_time)

        return f"{min_time:.2f} hours"
    except Exception as e:
        return f"Error: {e}"

# Streamlit app UI
st.title("Time Prediction")

# Button to train the model and predict time
if st.button("Predict Time"):
    # Train the model (if needed) and predict time
    with st.spinner("Training the model and making predictions..."):
        #train_model(training_file_path)  # Train the model (use predefined training data)
        result = predict_time(test_file_path)  # Predict time using predefined test data
    
    st.write(f"Predicted Time to Breakdown: {result}")
    st.success("Prediction complete!")








#################### Classification    ###############################

import streamlit as st
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib



# Predefined file paths for the model and scaler
#model_path = os.path.join(model_folder_path, 'cbm_breakdown_prediction_model.h5')
#scaler_path = os.path.join(model_folder_path, 'scalerFINP.pkl')

# Function to set random seed for reproducibility
def set_random_seed(seed=42):
    np.random.seed(seed=42)

# Define the training function
def train_model_classification(training_file_path):
    def load_data(file_path):
        df = pd.read_excel(file_path, sheet_name="Classification")
        X = df.iloc[:, 3:-1].values  # Assuming features are from column 1 to second last
        y = df.iloc[:, -1].values  # Target is in the last column
        return X, y

    def preprocess_data(X, y):
        # Scale the input features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Save the scaler for future use
        #joblib.dump(scaler, scaler_path)
        joblib.dump(scaler, os.path.join(model_folder_path, "scalerFINP.pkl"))
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.01, random_state=42)
        return X_train, X_val, y_train, y_val

    def build_model(input_shape):
        # Build the neural network model
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(4, activation='softmax'))  # 4 output units for the 4 classes
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
        return model

    # Set random seed for reproducibility
    set_random_seed()

    # Load and preprocess the data
    X, y = load_data(training_file_path)
    X_train, X_val, y_train, y_val = preprocess_data(X, y)
    
    class_weight_nn = {0: 1.0, 1: 500, 2: 60, 3: 60}

    # Build the model
    model = build_model(X_train)

    # Set up early stopping
    #early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100,     batch_size=32,class_weight=class_weight_nn)

    # Save the trained model
    #model.save(model_path)
    # model = joblib.load(os.path.join(model_folder_path, 'ensemble_modelFINP.pkl'))
    model.save(os.path.join(model_folder_path, 'ensemble_modelFINP.h5'))

    # joblib.dump(model, os.path.join(model_folder_path, 'ensemble_modelFINP.pkl'))
    st.success("Model training completed and saved!")

# Define the prediction function
def predict_breakdown(test_file_path):
    def load_test_data(file_path):
        df = pd.read_excel(file_path)
        X_test = df.iloc[:, 3:-1].values  # Features from column 1 to second last
        return df, X_test

    def preprocess_test_data(X_test):
        # Load the scaler and transform the test data
        #scaler = joblib.load(scaler_path)
        scaler = joblib.load(os.path.join(model_folder_path, "scalerFINP.pkl"))
        X_test_scaled = scaler.transform(X_test)
        return X_test_scaled

    def predict_classification(X_test_scaled):
        # Load the trained model and make predictions
        #model = load_model(model_path)
        model = load_model(os.path.join(model_folder_path, 'ensemble_modelFINP.h5'))
       # model = joblib.load(model, os.path.join(model_folder_path, 'ensemble_modelFINP.pkl'))
        predictions = model.predict(X_test_scaled)
        return predictions

    # Set random seed for reproducibility
    set_random_seed()

    try:
        # Load and preprocess the test data
        df, X_test = load_test_data(test_file_path)
        X_test_scaled = preprocess_test_data(X_test)

        # Make predictions
        predictions = predict_classification(X_test_scaled)

        # Get the predicted class labels (highest probability class)
        predicted_classes = np.argmax(predictions, axis=1)

        # Map the predicted classes to breakdown codes (0, 1, 2, 3)
        breakdown_codes = ["Code 0", "Code 1", "Code 2", "Code 3"]
        predicted_labels = [breakdown_codes[i] for i in predicted_classes]

        # Check if any non-zero breakdown code (Code 1, 2, or 3) is predicted
        non_zero_codes = [code for code in predicted_labels if "Code 1" in code or "Code 2" in code or "Code 3" in code]

        # Only return results if non-zero codes are predicted
        # if non_zero_codes:
        #     df['Predicted Breakdown'] = predicted_labels
        #     return df[['Predicted Breakdown']], "Non-zero breakdown codes predicted!"
        # else:
        #     # Return a message indicating no breakdown was predicted
        #     return None, "No BD predicted"

                # Only return results if non-zero codes are predicted
        if non_zero_codes:
            unique_non_zero_codes = set(non_zero_codes)
            num_unique_non_zero_codes = len(unique_non_zero_codes)
            df['Predicted Breakdown'] = predicted_labels
            return (
                df[['Predicted Breakdown']], 
                f"Non-zero breakdown codes predicted! Count: {num_unique_non_zero_codes}, Codes: {', '.join(unique_non_zero_codes)}"
            )
        else:
            # Return a message indicating no breakdown was predicted
            return None, "No BD predicted"
    except Exception as e:
        return f"Error: {e}", None

# Streamlit app UI
st.title("Breakdown Code Classification")


if st.button("check BD classification"):
    # Train the model (if needed) and predict time
    with st.spinner("Training the model and making predictions..."):
        #train_model_classification(training_file_path)  # Train the model (use predefined training data)
        result = predict_breakdown(test_file_path)  # Predict time using predefined test data
    
    st.write(f"classified breakdown: {result}")
    st.success("Prediction complete!")
