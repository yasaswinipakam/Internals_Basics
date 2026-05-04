import argparse
import joblib
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--wind_speed_kmph", type=float, required=True)
parser.add_argument("--humidity_pct", type=float, required=True)
parser.add_argument("--payload_mass_kg", type=float, required=True)
parser.add_argument("--vehicle_type_index", type=int, required=True)

args = parser.parse_args()

# Load model
model = joblib.load("models/tuned_model.pkl")

# Prepare input
X = np.array([[
    args.wind_speed_kmph,
    args.humidity_pct,
    args.payload_mass_kg,
    args.vehicle_type_index
]])

# Predict
prediction = model.predict(X)[0]

print(prediction)
