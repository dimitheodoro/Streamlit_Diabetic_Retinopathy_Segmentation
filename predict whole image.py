from skimage.io import  imread
import numpy as np
import pandas as pd
import os
import cv2
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from tqdm import tqdm 
from skimage import io, util
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from matplotlib.pyplot import figure

image_path = '/content/drive/MyDrive/IDRiD_17.jpg'
original=imread(image_path)
# os.mkdir(r'C:\Users\User\Desktop\Patched Image')
# os.mkdir(r'C:\Users\User\Desktop\Predicted Patches')
model_MA = load_model('/content/drive/MyDrive/Weights of UNET/MA/Mobilenet_dice.h5',compile=False)
model_EX = load_model('/content/drive/MyDrive/Weights of UNET/EX/Mobilenet_dice.h5',compile=False)
model_HM = load_model('/content/drive/MyDrive/Weights of UNET/HM/Mobilenet_dice.h5',compile=False)
model_SE = load_model('/content/drive/MyDrive/Weights of UNET/SE/Mobilenet_dice.h5',compile=False)


def create_patches(image_path, patch_width, patch_height,model,lesion):
    
    
    im1=cv2.imread(image_path)
    im1=cv2.cvtColor(im1,cv2.COLOR_BGR2RGB)
    x1,y1=im1.shape[0]//patch_width,im1.shape[1]//patch_width
    x2,y2=x1*patch_width,y1*patch_width
    im3 = cv2.resize(im1, (y2,x2))  


    patches = util.view_as_blocks(im3, (patch_width, patch_height,3))
    print("shape:::::::::::",patches.shape)
    predictionsj=[]
    predictions=[]
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i, j]
            

            # test_image = image.load_img(patch, target_size = (40, 40,3)) 
            # test_image = image.img_to_array(test_image)
            # test_image = np.expand_dims(patch, axis = 0)
            predict0=model.predict(patch/255)
           
            predict0=predict0[0]
            predict0=(predict0>0.5)*255
            predict0=np.concatenate([predict0,predict0,predict0],axis=2)

            mask=np.all(predict0==[255,255,255],axis=2)
            if lesion=='EX':
              predict0[mask]=[255,0,255]
            if lesion=='HM':
              predict0[mask]=[0,255,255]
            if lesion=='MA':
              predict0[mask]=[0,0,255]
            if lesion=='SE':
              predict0[mask]=[255,255,0]          

            predictionsj.append(predict0)



    predictions.append(predictionsj)    
    predictions=np.asarray(predictions)
    predictions=(predictions>0.5)*255
    predictions=predictions.reshape(x1,y1,1,patch_width,patch_width,3)   


            
    return predictions,x1,x2,y1,y2,im3
    


patches_EX,ex1,ex2,ey1,ey2,rim3=create_patches(image_path,512,512,model_EX,'EX')
patches_HM,hx1,hx2,hy1,hy2,him3=create_patches(image_path,512,512,model_HM,'HM')
patches_MA,mx1,mx2,my1,my2,mim3=create_patches(image_path,512,512,model_MA,'MA')
patches_SE,sx1,sx2,sy1,sy2,sim3=create_patches(image_path,512,512,model_SE,'SE')



def reconstruct_image(patches,x2,y2):
    print(patches.shape)
    img_height = x2
    img_width =y2
    print("patches before transpose",patches.shape)
    z=patches.transpose(0, 3, 2, 1, 4, 5).reshape(img_height, img_width,3)
    return z
    print("patches after transpose",z.shape)


reconstructed_EX=reconstruct_image(patches_EX,ex2,ey2)
reconstructed_HM=reconstruct_image(patches_HM,hx2,hy2)
reconstructed_MA=reconstruct_image(patches_MA,mx2,my2)
reconstructed_SE=reconstruct_image(patches_SE,sx2,sy2)



reconstructed_EX=np.squeeze(reconstructed_EX)
reconstructed_HM=np.squeeze(reconstructed_HM)
reconstructed_MA=np.squeeze(reconstructed_MA)
reconstructed_SE=np.squeeze(reconstructed_SE)

figure(figsize=(15,15),dpi=80)
plt.imshow(reconstructed_EX+reconstructed_HM+reconstructed_MA+reconstructed_SE)


plt.show()
figure(figsize=(9,9),dpi=80)
plt.imshow(original)
plt.show()