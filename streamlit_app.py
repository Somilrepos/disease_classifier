import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation,Dropout, MaxPooling2D,BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, load_model
import streamlit as st
import cv2
import numpy as np
import pandas as pd

# Defining the model
model_name='EfficientNetB3'
base_model=tf.keras.applications.EfficientNetB2(include_top=False, weights="imagenet",input_shape=(300,300,3), pooling='max')
x=base_model.output
x=keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)
x = Dense(256, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),
                bias_regularizer=regularizers.l1(0.006) ,activation='relu')(x)
x=Dropout(rate=.45, seed=123)(x)
output=Dense(10, activation='softmax')(x)
model=Model(inputs=base_model.input, outputs=output)
model.compile(Adamax(lr=.001), loss='categorical_crossentropy', metrics=['accuracy'])

model = load_model("EfficientNetB3-skin disease-86.70.h5")

# Image uploader
file = st.file_uploader("Upload a file")
if file:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.resize(image, (300,300))
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)
    category_idx = np.argmax(pred)
    df = pd.read_csv("./class_dict.csv")
    category = df.loc[category_idx,"class"]
    
    st.write(f"Your prediction is {category} with a accuracy of {(pred[0][category_idx]*100):.2f}")
    
