from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Charger le modèle
model = joblib.load('svm_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['embedding']  # Un tableau de 4096 valeurs par exemple
    prediction = model.predict([np.array(data)])
    return jsonify({'result': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
