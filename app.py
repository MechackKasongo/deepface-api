from flask import Flask, request, jsonify
import joblib
from deepface import DeepFace
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Charger le modèle SVM
model = joblib.load("svm_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Vérifier si une image est envoyée
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image envoyée'}), 400

    file = request.files['image']

    # 2. Lire le fichier image en mémoire
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    # 3. Utiliser DeepFace pour extraire l'embedding
    try:
        embedding = DeepFace.represent(img_path=np.array(img), model_name='VGG-Face')[0]["embedding"]
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # 4. Prédiction avec le modèle SVM
    prediction = model.predict([embedding])[0]

    return jsonify({'class': prediction})



# from flask import Flask, request, jsonify
# import joblib
# import numpy as np
# import os

# app = Flask(__name__)

# # Charger le modèle
# model = joblib.load('svm_model.pkl')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json['embedding']  # Un tableau de 4096 valeurs par exemple
#     prediction = model.predict([np.array(data)])
#     return jsonify({'result': prediction[0]})

if __name__ == '__main__':
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
