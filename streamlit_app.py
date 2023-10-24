import onnxruntime as rt
import streamlit as st
import cv2
import numpy as np
import pandas as pd

category = None

def load(model_name):
	  return rt.InferenceSession(model_name)

sess = load("model.onnx")

# Image uploader
file = st.file_uploader("Upload a file")
if file:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.resize(image, (300,300))
    image = np.expand_dims(image, axis=0)
    pred = sess.run(['dense_1'], {'input': image})
    category_idx = np.argmax(pred)
    df = pd.read_csv("./class_dict.csv")
    category = df.loc[category_idx,"class"]
    print(pred[0][0])
    st.write(f"Your prediction is {category} with a accuracy of {(pred[0][0][category_idx]*100):.2f}")
    pred_df = pd.DataFrame({'Disease Class': df.loc[:,'class'], 'Prediction Probabilities (in %)': np.multiply(pred[0][0],100)}, columns = ['Disease Class','Prediction Probabilities (in %)'])
    st.write(pred_df)


