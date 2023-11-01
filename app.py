{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7d3315c5-41b8-4c47-b2f6-08fae3730359",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "import os\n",
    "import shutil\n",
    "import random\n",
    "import cv2\n",
    "import keras\n",
    "import matplotlib.pyplot as plt\n",
    "# Importing tensorflow and its utilities\n",
    "from tensorflow.keras.metrics import categorical_crossentropy\n",
    "from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input\n",
    "from tensorflow.keras.applications.inception_v3 import InceptionV3\n",
    "from tensorflow.keras.applications.resnet50 import ResNet50\n",
    "from tensorflow.keras.applications.densenet import DenseNet169, DenseNet201\n",
    "from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B1,EfficientNetV2B2,EfficientNetV2B3\n",
    "from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau\n",
    "from tensorflow.keras.models import load_model\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img\n",
    "from sklearn.model_selection import train_test_split\n",
    "from keras.layers import Dense, Activation, GlobalAveragePooling2D, Dropout, Flatten\n",
    "from keras.applications.vgg19 import VGG19, preprocess_input\n",
    "from keras.preprocessing import image\n",
    "from keras.preprocessing.image import ImageDataGenerator\n",
    "from keras.models import Model\n",
    "from keras.optimizers import Adam,Adagrad\n",
    "from glob import glob"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "1383d75b-8202-4fe0-9a68-7349ff59b5a4",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "from tensorflow.keras.models import load_model\n",
    "new=load_model(r'D:\\VIT\\Area of Research\\Computer Vision\\Nutrient Deficiency\\Dataset Refer\\My execution\\jupyter\\work1\\Bdata\\Mobilenasensemble.hd5', compile = False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "98d1c020-735a-4087-84a0-2836a5861cde",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on http://127.0.0.1:5000\n",
      "Press CTRL+C to quit\n",
      "127.0.0.1 - - [01/Nov/2023 10:57:13] \"GET / HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1/1 [==============================] - 7s 7s/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [01/Nov/2023 10:57:27] \"POST / HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [01/Nov/2023 10:57:27] \"GET /static/b.0.jpg HTTP/1.1\" 200 -\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask,redirect, url_for, request, render_template\n",
    "from werkzeug.utils import secure_filename\n",
    "from gevent.pywsgi import WSGIServer\n",
    "from tensorflow.keras.preprocessing.image import load_img\n",
    "from tensorflow.keras.preprocessing.image import img_to_array\n",
    "#from tensorflow.keras.applications.vgg16 import preprocess_input\n",
    "#from tensorflow.keras.applications.vgg16 import decode_predictions\n",
    "#from keras.applications.vgg16 import VGG16\n",
    "import json\n",
    "app = Flask(__name__)\n",
    "\n",
    "@app.route(\"/\", methods=['GET'])\n",
    "\n",
    "def index():\n",
    "    return render_template(\"index.html\")\n",
    "\n",
    "@app.route('/', methods=['GET','POST'])\n",
    "def predict():\n",
    "    if request.method == 'POST':\n",
    "        imagefile= request.files['imagefile']\n",
    "        image_path = \"static/\" + imagefile.filename\n",
    "        imagefile.save(image_path)\n",
    "        image = load_img(image_path, target_size=(224, 224))\n",
    "        image = img_to_array(image)\n",
    "        image = np.expand_dims(image, axis=0)\n",
    "        yhat = new.predict(image)\n",
    "        label = decode_predictions(yhat)\n",
    "        label = label[0][0]\n",
    "    classification = '%s (%.2f%%)' % (label[1], label[2]*100)\n",
    "    return render_template(\"index.html\", prediction=classification, image_path=image_path)\n",
    " \n",
    "    \n",
    "def decode_predictions(yhat, top=4, class_list_path=r'C:\\Users\\M.Sudhakar\\anaconda3\\Scripts\\deploy\\templates\\index.json'):\n",
    "    if len(yhat.shape) != 2 or yhat.shape[1] != 4: # your classes number\n",
    "        raise ValueError('`decode_predictions` expects '\n",
    "                     'a batch of predictions '\n",
    "                    '(i.e. a 2D array of shape (samples, 1000)). '\n",
    "                   'Found array with shape: ' + str(yhat.shape))\n",
    "    index_list = json.load(open(class_list_path))\n",
    "    results = []\n",
    "    for pred in yhat:\n",
    "            top_indices = pred.argsort()[-top:][::-1]\n",
    "            result = [tuple(index_list[str(i)]) + (pred[i],)for i in top_indices]\n",
    "            result.sort(key=lambda x: x[2], reverse=True)\n",
    "            results.append(result)\n",
    "    return results\n",
    "    \n",
    "if __name__ == '__main__':\n",
    "    app.run(port =5000, debug=False)"
   ]
  },
  
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "tf",
   "language": "python",
   "name": "tf"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.11"
  },
  "toc-autonumbering": true,
  "toc-showmarkdowntxt": true,
  "toc-showtags": true
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
