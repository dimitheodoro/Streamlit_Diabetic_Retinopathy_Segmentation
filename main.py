
# from keras.models import load_model
# from keras.preprocessing import image
# import numpy as np
# # import matplotlib.pyplot as plt
# from  PIL import *
# import streamlit as st
# import cv2
# from skimage.io import imread
# from skimage import io, util
# # from io import StringIO

# FORTH = imread('e17_ventures.jpg')
# st.header("Segmentation of Diabetic Retinopathy lesions")
# st.subheader("by Theodoropoulos Dimitrios")
# st.image(FORTH)

# Exudates='Test images/Exudates.png'
# Hemmorhages='Test images/Hemmorhages.png'
# Microaeurysms='Test images/Microaeurysms.png'
# Soft_Exudates='Test images/Soft_Exudates.png'
# images={'Exudates':Exudates,'Hemmorhages':Hemmorhages,'Microaeurysms':Microaeurysms,'Soft Exudates':Soft_Exudates}

# def load_model_(Original_image,lesion):  
#     model = load_model(lesion,compile=False)
#     test_image0= image.load_img(Original_image, target_size = (512, 512,3)) 
#     test_image = image.img_to_array(test_image0)
#     test_image=test_image/255.0
#     test_image = np.expand_dims(test_image, axis = 0)
#     predict=model.predict(test_image)
#     predict=predict[0]
#     predict=np.concatenate([predict,predict,predict],axis=2)
#     predict=(predict>0.5)*255
#     return test_image0,predict

# def load_model_2(Original_image,lesion):  
#     model = load_model(lesion,compile=False)
#     test_image0 =Image.open(Original_image)
#     newsize = (512, 512)
#     test_image = test_image0.resize(newsize)
#     test_image_array= image.img_to_array(test_image)
#     test_image=test_image_array/255.0
#     test_image = np.expand_dims(test_image, axis = 0)
#     predict=model.predict(test_image)
#     predict=predict[0]
#     predict=np.concatenate([predict,predict,predict],axis=2)
#     predict=(predict>0.5)*255
#     return test_image_array,predict

# radiobox = st.sidebar.radio(
#      "Choose the type of lesion to be segmented",
#      ('Exudates', 'Hemmorhages', 'Microaeurysms','Soft Exudates'))

# selectbox = st.sidebar.selectbox(
#      "If you want a demo form our images choose demo ,or else upload your fundus image",
#      ('demo', 'Segment my full retinal image',))

# if selectbox == 'demo':
#     st.title(images[radiobox][12::][:-4])
#     col1,col2 = st.beta_columns(2)
#     resized_image = cv2.resize(imread( images[radiobox]),None,fx=0.5,fy=0.5)
#     with col1:
#         st.image(resized_image,caption='original image')
#     with col2:
#         _,prediction = load_model_(images[radiobox],images[radiobox][12::][:-4]+'_weights.h5')
#         new_dims = (resized_image.shape[0],resized_image.shape[1])
#         resized_prediction = cv2.resize(prediction.astype('float32'),new_dims)
#         st.image(resized_prediction,clamp=True,caption='segmented image')


        
# if selectbox == 'Segment my full retinal image':
#     image_path = st.file_uploader("Choose a fundus image from your devise",type=['png', 'jpg'])
#     st.write("Exudates are magenta, Hemmorhages are cyan ,Microaeurysms are blue ,Soft Exudates are yellow")
#     if image_path is not None:
#         col1,col2 = st.beta_columns(2)
#         # image_path = 'Test images\IDRiD_022.jpg'
#         # original=imread(image_path)

#         model_MA = load_model('Microaeurysms_weights.h5',compile=False)
#         model_EX = load_model('Exudates_weights.h5',compile=False)
#         model_HM = load_model('Hemmorhages_weights.h5',compile=False)
#         model_SE = load_model('Soft_Exudates_weights.h5',compile=False)


#         def create_patches(image_path, patch_width, patch_height,model,lesion):
#             global im3
#             im1=Image.open(image_path)
#             im1=np.array(im1)
#             # st.write("Please wait while procceding...")
#             # im1=cv2.cvtColor(im1,cv2.COLOR_BGR2RGB)
#             x1,y1=im1.shape[0]//patch_width,im1.shape[1]//patch_width
#             x2,y2=x1*patch_width,y1*patch_width
#             im3 = cv2.resize(im1, (y2,x2))  
            

#             patches = util.view_as_blocks(im3, (patch_width, patch_height,3))

#             predictionsj=[]
#             predictions=[]
#             for i in range(patches.shape[0]):
#                 for j in range(patches.shape[1]):

#                     patch = patches[i, j]
                    
#                     predict0=model.predict(patch/255)
                
#                     predict0=predict0[0]
#                     predict0=(predict0>0.5)*255
#                     predict0=np.concatenate([predict0,predict0,predict0],axis=2)

#                     mask=np.all(predict0==[255,255,255],axis=2)
#                     if lesion=='EX':
#                         predict0[mask]=[255,0,255]
#                     if lesion=='HM':
#                         predict0[mask]=[0,255,255]
#                     if lesion=='MA':
#                         predict0[mask]=[0,0,255]
#                     if lesion=='SE':
#                         predict0[mask]=[255,255,0]          

#                     predictionsj.append(predict0)


#             predictions.append(predictionsj)    
#             predictions=np.asarray(predictions)
#             predictions=(predictions>0.5)*255
#             # predictions=predictions.reshape(x1,y1,1,patch_width,patch_width,3)   
#             predictions=predictions.reshape(x1,y1,1,patch_width,patch_width,3)   
        
#             return predictions,x1,x2,y1,y2,im3
            
#         patches_EX,ex1,ex2,ey1,ey2,rim3=create_patches(image_path,512,512,model_EX,'EX')
#         patches_HM,hx1,hx2,hy1,hy2,him3=create_patches(image_path,512,512,model_HM,'HM')
#         patches_MA,mx1,mx2,my1,my2,mim3=create_patches(image_path,512,512,model_MA,'MA')
#         patches_SE,sx1,sx2,sy1,sy2,sim3=create_patches(image_path,512,512,model_SE,'SE')


#         def reconstruct_image(patches,x2,y2):
#             print(patches.shape)
#             img_height = x2
#             img_width =y2
#             print("patches before transpose",patches.shape)
#             z=patches.transpose(0, 3, 2, 1, 4, 5).reshape(img_height, img_width,3)
#             return z
            

#         reconstructed_EX=reconstruct_image(patches_EX,ex2,ey2)
#         reconstructed_HM=reconstruct_image(patches_HM,hx2,hy2)
#         reconstructed_MA=reconstruct_image(patches_MA,mx2,my2)
#         reconstructed_SE=reconstruct_image(patches_SE,sx2,sy2)

#         reconstructed_EX=np.squeeze(reconstructed_EX)
#         reconstructed_HM=np.squeeze(reconstructed_HM)
#         reconstructed_MA=np.squeeze(reconstructed_MA)
#         reconstructed_SE=np.squeeze(reconstructed_SE)

#         reconstructed = reconstructed_EX+reconstructed_HM+reconstructed_MA+reconstructed_SE


#         with col1:
#             st.image(im3,caption='original image')
#         with col2:
#             st.image(reconstructed,caption='segmented image',clamp=True)




##################################################################################### OLD CODE ##########
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
# import matplotlib.pyplot as plt
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




