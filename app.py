from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("crop_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Input features
    features = [[
        data["N"],
        data["P"],
        data["K"],
        data["temperature"],
        data["humidity"],
        data["ph"],
        data["rainfall"]
    ]]

    # Prediction
    crop = model.predict(features)[0]

    # Confidence
    confidence = max(model.predict_proba(features)[0])

    return jsonify({
        "crop": crop,
        "confidence": round(confidence * 100, 2),
        "profit": "₹30,000 - ₹60,000 per acre",
        "tip": "Suitable crop based on soil & climate"
    })

if __name__ == "__main__":
   if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)