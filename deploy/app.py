from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
model = load_model('Model_TrashCash_V3.h5')
threshold = 0.8  # Adjust this threshold value as needed

@app.route('/')
def serve_index():
    with open('index.html', 'r') as file:
        return file.read()

@app.route('/classify', methods=['POST'])
def classify():
    file = request.files['file']
    image = Image.open(file).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)

    result = model.predict(image_batch)
    class_index = np.argmax(result)
    confidence = result[0][class_index]
    classes = ['sellable', 'unsellable']
    classification = classes[class_index]

    if confidence < threshold:
        classification = 'unknown'

    response = {'result': classification}
    return jsonify(response)

if __name__ == '__main__':
    app.run()
