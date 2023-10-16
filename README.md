# Image-Caption-Generator

You saw an image and your brain can easily tell what the image is about, but can a computer tell what the image is representing? This is what we are going to implement in this Python based project where we will use deep learning techniques of Convolutional Neural Networks and a type of Recurrent Neural Network (LSTM) together.

# What is Image Caption Generator?
Image caption generator is a task that involves computer vision and natural language processing concepts to recognize the context of an image and describe them in a natural language like English.

# Dataset
flickr dataset link :- https://www.kaggle.com/datasets/adityajn105/flickr8k

Dataset consisting of 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events.

# What is CNN?
Convolutional Neural networks are specialized deep neural networks which can process the data that has input shape like a 2D matrix. Images are easily represented as a 2D matrix and CNN is very useful in working with images. It scans images from left to right and top to bottom to pull out important features from the image and combines the feature to classify images.

# What is LSTM?
LSTM stands for Long short term memory, they are a type of RNN (recurrent neural network) which is well suited for sequence prediction problems. Based on the previous text, we can predict what the next word will be. It has proven itself effective from the traditional RNN by overcoming the limitations of RNN which had short term memory. LSTM can carry out relevant information throughout the processing of inputs and with a forget gate, it discards non-relevant information.

# Image Caption Generator Model
to make our image caption generator model, we will be merging these architectures. It is also called a CNN-RNN model.

CNN is used for extracting features from the image. We will use the pre-trained model VGG16.
LSTM will use the information from CNN to help generate a description of the image.

#### Model :- https://www.kaggle.com/code/ajr094/image-caption-generator/output?select=best_model.h5
#### Kaggle NoteBook :- https://www.kaggle.com/code/ajr094/image-caption-generator/notebook
