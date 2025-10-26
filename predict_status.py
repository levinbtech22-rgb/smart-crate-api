import joblib
import pandas as pd
from flask import Flask, request, jsonify
import warnings
import os # Import os module

# Suppress specific warnings from joblib/sklearn if they occur during loading/prediction
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning) # Ignore potential future Pandas warnings

# --- Define Condition Rules (Must match training script) ---
TEMP_BAD_LOW = 12.0
TEMP_BAD_HIGH = 25.0
HUMIDITY_BAD_HIGH = 95.0
VOC_BAD_LOW = 150.0 # High VOCs = low resistance reading (kOhm)

# --- Determine File Paths Relative to the Script ---
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'tomato_status_classifier.joblib')
encoder_path = os.path.join(script_dir, 'label_encoder.joblib')


# --- Load the Trained Model and Label Encoder ---
# Initialize variables to None in case loading fails
model = None
label_encoder = None
try:
    # Ensure these files are in the same directory when deployed
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    print("--- API Startup: Model and label encoder loaded successfully. ---")
except FileNotFoundError:
    print("\nFATAL ERROR: Model or label encoder file not found on startup.")
    print(f"Looked for model at: {model_path}")
    print(f"Looked for encoder at: {encoder_path}")
    print("Make sure files are in the same directory as this script.")
    # In a real deployment, you might want more robust error handling or logging
except Exception as e:
    print(f"\nFATAL ERROR loading model/encoder on startup: {e}")

# --- Create Flask App ---
app = Flask(__name__)

# --- Prediction Function (Hybrid: Rules + ML) ---
def get_prediction(temperature, humidity, voc):
    """
    Predicts tomato status using rules first, then the ML model.
    Returns the predicted status string ('Good', 'Moderate', 'Bad', 'Error').
    """
    # Check if model and encoder loaded properly during startup
    if model is None or label_encoder is None:
        print("Error: Model or encoder not loaded. Cannot predict.")
        return "Error: Model not loaded"

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
        prediction_encoded = model.predict(input_data)[0]

        # Decode the prediction back to the original label ('Good' or 'Moderate')
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
        return prediction_label

    except Exception as e:
        print(f"Error during ML prediction: {e}")
        # Fallback in case of prediction error (e.g., unexpected input)
        return "Error: Prediction failed" # Return specific error

# --- API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    API endpoint to receive sensor data and return status prediction.
    Expects JSON input like:
    {
        "temperature": 18.5,
        "humidity": 90.1,
        "voc": 350.5
    }
    Returns JSON like:
    {"predicted_status": "Moderate"}
    or
    {"error": "Error message"}
    """
    # Check if model loaded correctly at startup
    if model is None or label_encoder is None:
         return jsonify({"error": "Model or Encoder not loaded on server."}), 500

    try:
        data = request.get_json()
        if not data:
             return jsonify({"error": "No JSON data received."}), 400

        # Validate presence of keys
        if 'temperature' not in data or 'humidity' not in data or 'voc' not in data:
             return jsonify({"error": "Missing required field(s). Expecting 'temperature', 'humidity', 'voc'."}), 400

        temp = data['temperature']
        hum = data['humidity']
        voc = data['voc']

        # Validate input types
        if not all(isinstance(val, (int, float)) for val in [temp, hum, voc]):
            return jsonify({"error": "Invalid input data types. Temperature, humidity, and voc must be numbers."}), 400

        # Get prediction using the hybrid function
        status = get_prediction(temp, hum, voc)

        # Check if prediction function returned an error
        if "Error" in status:
             return jsonify({"error": status}), 500

        # Return the successful prediction as JSON
        return jsonify({"predicted_status": status})

    except Exception as e:
        # Catch any other unexpected errors during request processing
        print(f"Error processing request: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

# --- Health Check Endpoint (Good practice for deployment) ---
@app.route('/', methods=['GET'])
def health_check():
    # Simple check to see if the API is running and model loaded
    if model is not None and label_encoder is not None:
        return jsonify({"status": "API is running and model loaded."})
    else:
        return jsonify({"status": "API is running BUT model/encoder failed to load."}), 500


# --- Run the App (Used for local testing, Render uses Gunicorn) ---
if __name__ == '__main__':
    # Make sure the server is accessible on your network for testing if needed
    # Port 10000 is often recommended by Render for free tier, or choose another like 5000
    app.run(host='0.0.0.0', port=10000, debug=False) # Turn debug off for safety