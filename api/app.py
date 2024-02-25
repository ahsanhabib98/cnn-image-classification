from tensorflow.keras.preprocessing.image import img_to_array, load_img
from flask import Flask, request, jsonify
import numpy as np
import pickle
import io

IMG_WIDTH = 150
IMG_HEIGHT = 150

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
labels = {0:'metal', 1:'plastic'}


# Define endpoint for prediction
@app.route('/api/image-label-predict', methods=['POST'])
def predict():
    # Check if image file is present in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})
    
    # Read the image file from the request
    file = request.files['image']
    
    # Open and preprocess the image
    img_bytes = io.BytesIO(file.read())  # Read file contents into BytesIO object
    img = load_img(img_bytes, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Get the predicted class label
    predicted_class = np.argmax(prediction)
    
    # Return the predicted label
    return jsonify({'predicted_label': labels[int(predicted_class)]})


if __name__ == '__main__':
    app.run()