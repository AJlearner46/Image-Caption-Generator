# Image-Caption-Generator

While humans can interpret these images without detailed captions, machines require some form of image captions for automatic understanding.

This project aims to develop an end-to-end solution for generating descriptive captions for images using deep learning techniques.

demo : https://image-captioin-generator-aj.streamlit.app/

## Dataset
- flickr dataset link :- https://www.kaggle.com/datasets/adityajn105/flickr8k
- I used the Flickr8k Dataset, which contains 8092 photographs and text descriptions. Dataset contain 5 caption for each Image

## Methodology for Image Captioning

### 1. Data Preprocessing
- Extract image features
- Text preprocessing
- Train-Test split
- Data generator

### 2. Encoder-Decoder Architecture
- Load VGG16 model
- Encoder : 
       Image feature layer
       Sequence feature layer
- Decoder
  ![image](https://github.com/AJlearner46/Image-Caption-Generator/assets/99804336/803e22b8-1536-40af-bf5a-06ca78e5c405)


### 3. Training & Optimization
- Training model
- Evaluation of model

### 4. Frontend
- User interface using streamlit.
 ![image](https://github.com/AJlearner46/Image-Caption-Generator/assets/99804336/5002201c-47b3-4946-90d4-8b9022590058)


## Results
- The VGG16-LSTM model was trained for 20 epochs, achieving a low training loss of 2.1828.
- I evaluated the model using the BLEU score, with a focus on BLEU-1 score (0.536631).
 ![image](https://github.com/AJlearner46/Image-Caption-Generator/assets/99804336/2629f4da-0290-4d1d-a0dc-d90135f1288f)




#### Model :- https://www.kaggle.com/code/ajr094/image-caption-generator/output?select=best_model.h5
#### Kaggle NoteBook :- https://www.kaggle.com/code/ajr094/image-caption-generator/notebook
