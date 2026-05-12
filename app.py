from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models and preprocessing artifacts
print("Loading models and preprocessing artifacts...")

# Load encoders and scaler
with open("models/ohe.pkl", "rb") as f:
    ohe = pickle.load(f)

with open("models/ordinal_encoder_stops.pkl", "rb") as f:
    ordinal_stops = pickle.load(f)

with open("models/ordinal_encoder_class.pkl", "rb") as f:
    ordinal_class = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load ANN model
ann_model = load_model("models/ann_model.h5", compile=False)


print("✓ All models loaded successfully")

# Dropdown options (based on training data)
OPTIONS = {
    "airline": ["SpiceJet", "AirAsia", "Vistara", "GO_FIRST", "Indigo", "Air_India"],
    "source_city": ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"],
    "destination_city": ["Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai", "Delhi"],
    "departure_time": ["Evening", "Early_Morning", "Morning", "Afternoon", "Night", "Late_Night"],
    "arrival_time": ["Night", "Morning", "Early_Morning", "Afternoon", "Evening", "Late_Night"],
    "stops": ["zero", "one", "two_or_more"],
    "class": ["Economy", "Business"]
}


def preprocess_input(form_data):
    """Preprocess user input to match training pipeline"""
    
    # Extract features
    airline = form_data["airline"]
    source_city = form_data["source_city"]
    destination_city = form_data["destination_city"]
    departure_time = form_data["departure_time"]
    arrival_time = form_data["arrival_time"]
    stops = form_data["stops"]
    flight_class = form_data["class"]
    duration = float(form_data["duration"])
    days_left = int(form_data["days_left"])
    
    # Create dataframe for encoding
    input_df = pd.DataFrame({
        "airline": [airline],
        "source_city": [source_city],
        "destination_city": [destination_city],
        "departure_time": [departure_time],
        "arrival_time": [arrival_time],
        "stops": [stops],
        "class": [flight_class],
        "duration": [duration],
        "days_left": [days_left]
    })
    
    # OneHot encode
    onehot_cols = ["airline", "source_city", "departure_time", "arrival_time", "destination_city"]
    ohe_encoded = ohe.transform(input_df[onehot_cols])
    ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out())
    
    # Ordinal encode
    stops_encoded = ordinal_stops.transform(input_df[["stops"]])[0][0]
    class_encoded = ordinal_class.transform(input_df[["class"]])[0][0]
    
    # Scale numerical features
    numerical_scaled = scaler.transform([[duration, days_left]])[0]
    
    # Combine all features
    final_input = pd.DataFrame({
        "duration": [numerical_scaled[0]],
        "days_left": [numerical_scaled[1]],
        "stops_encoded": [stops_encoded],
        "class_encoded": [class_encoded]
    })
    
    # Add one-hot encoded features
    final_input = pd.concat([final_input, ohe_df], axis=1)
    
    return final_input


@app.route("/")
def index():
    """Render the home page"""
    return render_template("index.html", options=OPTIONS, form_data={})


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Preprocess input
        input_processed = preprocess_input(form_data)
        
        # Make prediction (log-transformed)
        prediction_log = ann_model.predict(input_processed, verbose=0)[0][0]
        
        # Inverse transform to get actual price
        predicted_price = np.expm1(prediction_log)
        
        # Prepare result
        result = {
            "predicted_price": float(predicted_price),
            "model_name": "Predicted using Artificial Neural Network"
        }
        
        return render_template(
            "index.html",
            options=OPTIONS,
            form_data=form_data,
            prediction=result
        )
        
    except Exception as e:
        error_message = f"Prediction failed: {str(e)}"
        return render_template(
            "index.html",
            options=OPTIONS,
            form_data=request.form.to_dict(),
            error=error_message
        )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)