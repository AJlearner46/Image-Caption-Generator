import numpy as np
import pickle
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model, load_model

#-----------------------------------------------------------------

# Load the saved model
# D:\document\data sc\project sem 3\image caption generator\best_model.h5
model = load_model('best_model.h5')

#-----------------------------------------------------------------
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)
#tokenizer = Tokenizer()
max_length = 35

#-----------------------------------------------------------------

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items() :
        if index == integer:
            return word
    return None

#-----------------------------------------------------------------

def predict_caption(model, image, tokenizer, max_length):
    # Add the start tag for the generation process
    in_text = 'startseq'
    # Iterate over the max length of the sequence
    for _ in range(max_length):
        # Encode the input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # Predict the next word using the saved model
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

#--------------------------------------------------------------------

vgg_model = VGG16()
# restructure the model
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

#--------------------------------------------------------------------

image_path = '1028205764_7e8df9a2ea.jpg'
# Load image
image = load_img(image_path, target_size=(224, 224))
# Convert image pixels to a numpy array
image = img_to_array(image)
# Reshape data for model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# Preprocess image for VGG
image = preprocess_input(image)
# Extract features
feature = vgg_model.predict(image, verbose=0)
# Predict from the loaded model
caption = predict_caption(model, feature, tokenizer, max_length )
print(caption)

#----------------------------------------------------------------------

