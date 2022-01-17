from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from  PIL import *
import streamlit as st
import cv2
from skimage.io import imread


# os.chdir(r'C:\Users\User\Desktop\FINAL TUNING\Stremlit_DR')
FORTH = imread('FORTH.png')
st.header("Segmentation of Diabetic Retinopathy lesions")
st.image(FORTH)

Exudates=r'Test images\Exudates.png'
Hemmorhages=r'Test images\Hemmorhages.png'
Microaeurysms=r'Test images\Microaneurysms.png'
Soft_Exudates=r'Test images\Soft Exudates.png'

images={'Exudates':Exudates,'Hemmorhages':Hemmorhages,'Microaeurysms':Microaeurysms,'Soft Exudates':Soft_Exudates}


def load_model(Original_image,lesion):  
    model = load_model(lesion+'_weights.h5',compile=False)
    test_image0= image.load_img(Original_image, target_size = (512, 512,3)) 
    test_image = image.img_to_array(test_image0)
    test_image=test_image/255.0
    test_image = np.expand_dims(test_image, axis = 0)
    predict=model.predict(test_image)
    predict=predict[0]
    predict=np.concatenate([predict,predict,predict],axis=2)
    predict=(predict>0.5)*255
    return test_image0,predict


radiobox = st.sidebar.radio(
     "Choose the type of lesion to be segmented",
     ('Exudates', 'Hemmorhages', 'Microaeurysms','Soft Exudates'))

selectbox = st.sidebar.selectbox(
     "If you want a demo form our images choose demo ,or else upload your fundus image",
     ('demo', 'upload my image',))


if selectbox == 'demo':
    st.title(images[radiobox][12::][:-4]+'_weights.h5')
    col1,col2 = st.beta_columns(2)
    resized_image = cv2.resize(imread( images[radiobox]),None,fx=0.5,fy=0.5)
    with col1:
        st.image(resized_image,caption='original image')
        
    with col2:
        # st.image(resized_image,caption='original image')
        _,prediction = load_model(images[radiobox],images[radiobox][12::][:-4]+'_weights.h5')
        resized_prediction = cv2.resize(prediction,None,fx=0.5,fy=0.5)
        st.image(resized_prediction,caption='segmented image')

if selectbox == 'upload my image':
    st.title(images[radiobox][12::][:-4]+'_weights.h5')
    uploaded_file = st.file_uploader("Choose a fundus image from your devise")
    col1,col2 = st.beta_columns(2)
    resized_image = cv2.resize(imread( images[radiobox]),None,fx=0.5,fy=0.5)
    with col1:
        st.image(resized_image,caption='original image')
    with col2:
        # st.image(resized_image,caption='original image')
        _,prediction = load_model(images[radiobox],images[radiobox][12::][:-4]+'_weights.h5')
        resized_prediction = cv2.resize(prediction,None,fx=0.5,fy=0.5)
        st.image(resized_prediction,caption='segmented image')







