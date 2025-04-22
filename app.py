from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Charger le modèle
model = joblib.load('svm_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['embedding']  # Un tableau de 4096 valeurs par exemple
    prediction = model.predict([np.array(data)])
    return jsonify({'result': prediction[0]})

if __name__ == '__main__':
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
