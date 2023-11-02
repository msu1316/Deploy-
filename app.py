#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import shutil
import random
import cv2
import keras
import matplotlib.pyplot as plt
# Importing tensorflow and its utilities
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet169, DenseNet201
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B1,EfficientNetV2B2,EfficientNetV2B3
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, GlobalAveragePooling2D, Dropout, Flatten
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam,Adagrad
from glob import glob


# In[2]:


from tensorflow.keras.models import load_model
new=load_model(r'D:\VIT\Area of Research\Computer Vision\Nutrient Deficiency\Dataset Refer\My execution\jupyter\work1\Bdata\Mobilenasensemble.hd5', compile = False)


# In[3]:


from flask import Flask,redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.applications.vgg16 import preprocess_input
#from tensorflow.keras.applications.vgg16 import decode_predictions
#from keras.applications.vgg16 import VGG16
import json
app = Flask(__name__)

@app.route("/", methods=['GET'])

def index():
    return render_template("index.html")

@app.route('/', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        imagefile= request.files['imagefile']
        image_path = "static/" + imagefile.filename
        imagefile.save(image_path)
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        yhat = new.predict(image)
        label = decode_predictions(yhat)
        label = label[0][0]
    classification = '%s (%.2f%%)' % (label[1], label[2]*100)
    return render_template("index.html", prediction=classification, image_path=image_path)
 
    
def decode_predictions(yhat, top=4, class_list_path=r'C:\Users\M.Sudhakar\anaconda3\Scripts\deploy\templates\index.json'):
    if len(yhat.shape) != 2 or yhat.shape[1] != 4: # your classes number
        raise ValueError('`decode_predictions` expects '
                     'a batch of predictions '
                    '(i.e. a 2D array of shape (samples, 1000)). '
                   'Found array with shape: ' + str(yhat.shape))
    index_list = json.load(open(class_list_path))
    results = []
    for pred in yhat:
            top_indices = pred.argsort()[-top:][::-1]
            result = [tuple(index_list[str(i)]) + (pred[i],)for i in top_indices]
            result.sort(key=lambda x: x[2], reverse=True)
            results.append(result)
    return results
    
if __name__ == '__main__':
    app.run(port =5000, debug=False)


# In[ ]:





# In[ ]:




