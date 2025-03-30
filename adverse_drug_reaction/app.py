from flask import Flask, request, jsonify, render_template
import joblib
from data_preprocessing import preprocess_text

app = Flask(__name__)

model = joblib.load('models/adr_model.pkl')
scaler = joblib.load('models/scaler.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text]).toarray()
    scaled_text = scaler.transform(vectorized_text)
    prediction = model.predict(scaled_text)
    return jsonify({'prediction': prediction[0]})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    feedback = data['feedback']
    with open('feedback.txt', 'a') as f:
        f.write(feedback + '\n')
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
