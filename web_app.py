import streamlit as st
import os
from io import BytesIO
import requests
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# https://drive.google.com/file/d/1JMpnVSZg48LjjonZEmrFghsVAfJkKEJ9/view?usp=drive_link
# https://drive.google.com/file/d/1JMpnVSZg48LjjonZEmrFghsVAfJkKEJ9/view?usp=sharing

# Function to load model from Google Drive
def load_model_from_drive(file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    model_content = BytesIO(response.content)
    model = tensorflow.keras.models.load_model(model_content)
    return model

# Extract file ID from the Google Drive link
drive_link = "https://drive.google.com/file/d/1JMpnVSZg48LjjonZEmrFghsVAfJkKEJ9/view?usp=sharing"
file_id = drive_link.split("/file/d/")[-1].split("/")[0]

# Load the model
model = load_model_from_drive(file_id)


# Load the saved model
# model_path = os.path.abspath('best_model.h5')
# model = load_model(model_path)


# Load the tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

vgg_model = VGG16()

# restructure the model
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Define functions for image processing and caption generation
def load_and_preprocess_image(image_path):
    # Load and preprocess the image for VGG
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    # Extract features
    feature = vgg_model.predict(image, verbose=0)
    return feature

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items() :
        if index == integer:
            return word
    return None

def generate_caption(model, image, tokenizer):
    # Add the start tag for the generation process
    in_text = 'startseq'
    max_length = 35

    # Iterate over the max length of the sequence
    for _ in range(max_length):
        # Encode the input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # Predict the next word using the model
        yhat = model.predict([image, sequence], verbose=0)
        # Get the index with the highest probability
        yhat = np.argmax(yhat)
        # Convert the index to a word
        word = idx_to_word(yhat, tokenizer)
        # Stop if the word is not found
        if word is None:
            break
        # Append the word as input for generating the next word
        in_text += " " + word
        # Stop if we reach the end tag
        if word == 'endseq':
            break

    return in_text

st.title("Image Caption Generator")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    feature = load_and_preprocess_image(uploaded_image)

    if st.button("Generate Caption"):
        caption = generate_caption(model, feature, tokenizer )

        st.write("Generated Caption:", caption)
