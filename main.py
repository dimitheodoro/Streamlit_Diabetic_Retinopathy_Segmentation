from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from  PIL import *
import streamlit as st
import cv2
from skimage.io import imread
# from io import StringIO

FORTH = imread('FORTH.png')
st.header("Segmentation of Diabetic Retinopathy lesions")
st.image(FORTH)

Exudates='Test images/Exudates.png'
Hemmorhages='Test images/Hemmorhages.png'
Microaeurysms='Test images/Microaeurysms.png'
Soft_Exudates='Test images/Soft_Exudates.png'
images={'Exudates':Exudates,'Hemmorhages':Hemmorhages,'Microaeurysms':Microaeurysms,'Soft Exudates':Soft_Exudates}

def load_model_(Original_image,lesion):  
    model = load_model(lesion,compile=False)
    test_image0= image.load_img(Original_image, target_size = (512, 512,3)) 
    test_image = image.img_to_array(test_image0)
    test_image=test_image/255.0
    test_image = np.expand_dims(test_image, axis = 0)
    predict=model.predict(test_image)
    predict=predict[0]
    predict=np.concatenate([predict,predict,predict],axis=2)
    predict=(predict>0.5)*255
    return test_image0,predict

def load_model_2(Original_image,lesion):  
    model = load_model(lesion,compile=False)
    test_image0 =Image.open(Original_image)
    newsize = (512, 512)
    test_image = test_image0.resize(newsize)
    test_image_array= image.img_to_array(test_image)
    test_image=test_image_array/255.0
    test_image = np.expand_dims(test_image, axis = 0)
    predict=model.predict(test_image)
    predict=predict[0]
    predict=np.concatenate([predict,predict,predict],axis=2)
    predict=(predict>0.5)*255
    return test_image_array,predict

radiobox = st.sidebar.radio(
     "Choose the type of lesion to be segmented",
     ('Exudates', 'Hemmorhages', 'Microaeurysms','Soft Exudates'))

selectbox = st.sidebar.selectbox(
     "If you want a demo form our images choose demo ,or else upload your fundus image",
     ('demo', 'upload my image',))

if selectbox == 'demo':
    st.title(images[radiobox][12::][:-4])
    col1,col2 = st.beta_columns(2)
    resized_image = cv2.resize(imread( images[radiobox]),None,fx=0.5,fy=0.5)
    with col1:
        st.image(resized_image,caption='original image')
    with col2:
        _,prediction = load_model_(images[radiobox],images[radiobox][12::][:-4]+'_weights.h5')
        new_dims = (resized_image.shape[0],resized_image.shape[1])
        resized_prediction = cv2.resize(prediction.astype('float32'),new_dims)
        st.image(resized_prediction,clamp=True,caption='segmented image')

if selectbox == 'upload my image':
    st.title(images[radiobox][12::][:-4])
    uploaded_file = st.file_uploader("Choose a fundus image from your devise",type=['png', 'jpg'])
    if uploaded_file is not None:
        col1,col2 = st.beta_columns(2)
        resized_image = cv2.resize(imread( uploaded_file),(256,256))
        with col1:
            st.image(resized_image,caption='original image')
        with col2:
            _,prediction = load_model_2(uploaded_file,images[radiobox][12::][:-4]+'_weights.h5')
            new_dims = (resized_image.shape[0],resized_image.shape[1])
            resized_prediction = cv2.resize(prediction.astype('float32'),new_dims)
            st.image(resized_prediction,clamp=True,caption='segmented image')




