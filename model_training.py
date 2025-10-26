import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
Tomato=pd.read_csv('Tomato.csv')
import joblib
import pandas as pd
import warnings

# Suppress specific warnings from joblib/sklearn if they occur
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- Define Condition Rules (Must match training script) ---
TEMP_BAD_LOW = 12.0
TEMP_BAD_HIGH = 25.0
HUMIDITY_BAD_HIGH = 95.0
VOC_BAD_LOW = 150.0 # High VOCs = low resistance reading

# --- Load the Trained Model and Label Encoder ---
# Initialize variables to None in case loading fails
model = None
label_encoder = None
try:
    model = joblib.load('tomato_status_classifier.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    print("Model and label encoder loaded successfully.")
except FileNotFoundError:
    print("\nError: Model or label encoder file not found.")
    print("Please ensure you have run the 'model_training.py' script successfully")
    print("and that 'tomato_status_classifier.joblib' and 'label_encoder.joblib'")
    print("are in the same directory as this script.")
    exit()
except Exception as e:
    print(f"Error loading model/encoder: {e}")
    exit()

# --- Prediction Function (Hybrid: Rules + ML) ---
# Ensure this function is defined BEFORE it's called in the main block
def predict_tomato_status(temperature, humidity, voc, loaded_model, loaded_encoder):
    """
    Predicts tomato status using rules first, then the ML model.
    Requires the loaded model and encoder as arguments.
    Returns the predicted status string ('Good', 'Moderate', 'Bad').
    """
    # Check if model and encoder loaded properly
    if loaded_model is None or loaded_encoder is None:
        print("Error: Model or encoder not loaded. Cannot predict.")
        return "Error"

    # 1. Apply Rules First for 'Bad' status
    if (temperature < TEMP_BAD_LOW or
        temperature > TEMP_BAD_HIGH or
        humidity > HUMIDITY_BAD_HIGH or
        voc < VOC_BAD_LOW):
        return "Bad"

    # 2. If not 'Bad' by rules, use ML model for 'Good'/'Moderate'
    try:
        # Prepare input data in the same format as training data
        input_data = pd.DataFrame([[temperature, humidity, voc]],
                                  columns=['Temperature_C', 'Humidity_RH', 'VOC_Level_kOhm'])

        # Make prediction (returns encoded label, e.g., 0 or 1)
        prediction_encoded = loaded_model.predict(input_data)[0]

        # Decode the prediction back to the original label ('Good' or 'Moderate')
        prediction_label = loaded_encoder.inverse_transform([prediction_encoded])[0]
        return prediction_label

    except Exception as e:
        print(f"Error during ML prediction: {e}")
        # Fallback in case of prediction error (e.g., unexpected input)
        # Since rules already filtered 'Bad', defaulting to 'Moderate' is safer.
        return "Moderate"

# --- Main part to get user input and predict ---
# This block runs only when the script is executed directly
if __name__ == '__main__':
    # Check again if model loading was successful before proceeding
    if model is None or label_encoder is None:
        print("Exiting due to model/encoder loading failure.")
        exit()

    print("\nEnter the current sensor readings:")

    while True:
        try:
            temp_input = float(input("Temperature (°C): "))
            break
        except ValueError:
            print("Invalid input. Please enter a number for temperature.")

    while True:
        try:
            hum_input = float(input("Humidity (%RH): "))
            break
        except ValueError:
            print("Invalid input. Please enter a number for humidity.")

    while True:
        try:
            # Assuming VOC is still kOhm from sensor
            voc_input = float(input("VOC Level (kOhm - lower value means higher VOCs): "))
            break
        except ValueError:
            print("Invalid input. Please enter a number for VOC level.")

    # Get the prediction - pass the loaded model and encoder to the function
    predicted_status = predict_tomato_status(temp_input, hum_input, voc_input, model, label_encoder)

    # Print the result
    print(f"\n---> Predicted Tomato Status: {predicted_status}")