from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

application = Flask(__name__)

# Load the ML model and vectorizer globally when the app starts
loaded_model = None
vectorizer = None

# Load the model and vectorizer at startup
with open('basic_classifier.pkl', 'rb') as fid:
    loaded_model = pickle.load(fid)

with open('count_vectorizer.pkl', 'rb') as vd:
    vectorizer = pickle.load(vd)

@application.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.json
    text = data.get('text', '')

    # Make prediction
    prediction = loaded_model.predict(vectorizer.transform([text]))[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000)



# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"text": "This is fake news"}'
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"text\": \"This is fake news\"}"