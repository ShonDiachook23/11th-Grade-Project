import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('model.h5')
class_labels = ['Cat', 'Dog']
IMG_SIZE = 60

def preprocess_image(image):
    image = image.convert('L')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image).reshape((IMG_SIZE, IMG_SIZE, 1)) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def main():
    st.title('Image Classifier')
    st.write('Upload an image and the model will classify it as a dog or a cat.')
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        image_array = preprocess_image(image)

        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_labels[predicted_class]
        st.write(f'Prediction: {predicted_label}')

# Run the app
if __name__ == '__main__':
    main()