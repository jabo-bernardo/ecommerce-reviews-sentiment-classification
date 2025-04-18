import json
from flask import Flask, request
from flask_cors import CORS

from predict import load_model, predict_reviews

app = Flask(__name__)
CORS(app);

model, vectorizer = load_model();

@app.route("/predict", methods=['POST'])
def predict_review():
	jsonData = request.get_json();

	content = jsonData['content'];

	predicted_reviews = predict_reviews([content], model, vectorizer).tolist();

	return json.dumps({"success": True, "response": predicted_reviews[0]})