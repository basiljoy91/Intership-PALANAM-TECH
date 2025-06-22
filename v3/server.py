from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load Keras model and tokenizer
model = tf.keras.models.load_model("sentiment_model.h5")
tokenizer = joblib.load("tokenizer.pickle")

# Function to preprocess and predict
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Tokenize and pad the input
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)  # Match training maxlen

    # Predict
    prediction = model.predict(padded)[0][0]  # e.g., binary classification
    sentiment = "Positive" if prediction > 0.6 else "Negative" if prediction < 0.4 else "Neutral"

    # Score: convert to 1–10 scale
    score = round(prediction * 9 + 1, 2)

    return jsonify({
        "sentiment": sentiment,
        "score": score
    })

if __name__ == '__main__':
    app.run(port=5001, debug=True)
