import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import keras
from glob import glob
import cv2
from lib import *
import os
import segmentation_models as sm
import tensorflow as tf


def to_DataFrame(dir_image, dir_mask):    
    listd=os.listdir(f"/content/data/{dir_image}")
    pat=f"/content/data/{dir_image}"
    im_path=sorted([(f"{pat}/{x}") for x in listd])
    listd2=os.listdir(f"/content/data/{dir_mask}")
    pat2=f"/content/data/{dir_mask}"
    mas_path=sorted([(f"{pat2}/{x}") for x in listd2])
    dic={}
    for i in range(len(mas_path)):
        mask = np.array(Image.open(mas_path[i]))
        rle_mask = encode_rle(mask)
        dic[im_path[i]]=rle_mask
    df=pd.DataFrame.from_dict(dic, orient='index')
    df = df.reset_index()
    df.columns = ['img', 'mask']
    return(df)

def keras_generator(gen_df, batch_size):
    while True:
        x_batch = []
        y_batch = []
        
        for i in range(batch_size):
            img_name, mask_rle = gen_df.sample(1).values[0]
            img = np.array(Image.open(img_name))
            mask = decode_rle(mask_rle)
            
            img = cv2.resize(img, (256, 256))
            mask = cv2.resize(mask, (256, 256))
            
            
            x_batch += [img]
            y_batch += [mask]

        x_batch = np.array(x_batch) / 255.
        y_batch = np.array(y_batch)

        yield x_batch, np.expand_dims(y_batch, -1)

def generate_data(df, batch_size):
    i = 0
    file_list = list(df['img'])
    mask_list = list(df['mask'])
    while True:
        image_batch = []
        mask_batch = []
        for b in range(batch_size):
            if i == len(file_list):
                i = 0
            img_name=file_list[i]
            mask_rle=mask_list[i]
            i += 1
            img = np.array(Image.open(img_name))
            mask = decode_rle(mask_rle)
            img = cv2.resize(img, (256, 256))
            mask = cv2.resize(mask, (256, 256))
            image_batch.append(img)
            mask_batch.append(mask)
        image_batch = np.array(image_batch) / 255.
        mask_batch = np.array(mask_batch)   
        yield  image_batch, np.expand_dims(mask_batch, -1)

def Plot_graph(history):    
    plt.plot(history.history['score'])
    plt.plot(history.history['val_score'])
    plt.title('Model score')
    plt.ylabel('Score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

def ExamplePred(model, dft):   
    for x, y in keras_generator(dft, 16):
        break
    yp=((model.predict(x))>0.5)
    print(y.shape)
    show_img_with_mask(x[0], yp[0,:,:,0])
    show_img_with_mask(x[1], yp[1,:,:,0])
    show_img_with_mask(x[2], yp[2,:,:,0])
    show_img_with_mask(x[3], yp[3,:,:,0])
    show_img_with_mask(x[4], yp[4,:,:,0])
    show_img_with_mask(x[5], yp[5,:,:,0])
    show_img_with_mask(x[6], yp[6,:,:,0])
    show_img_with_mask(x[7], yp[7,:,:,0])
    show_img_with_mask(x[8], yp[8,:,:,0])

def ExampleBestPred(file_h5, encoder_freeze=False):
    BACKBONE='Resnet34'
    model=sm.Unet(BACKBONE, encoder_weights='imagenet', encoder_freeze=encoder_freeze)
    model.load_weights(file_h5)
    for x, y in keras_generator(dft, 16):
        break
    yp=((model.predict(x))>0.5)
    print(y.shape)
    show_img_with_mask(x[0], yp[0,:,:,0])
    show_img_with_mask(x[1], yp[1,:,:,0])
    show_img_with_mask(x[2], yp[2,:,:,0])
    show_img_with_mask(x[3], yp[3,:,:,0])
    show_img_with_mask(x[4], yp[4,:,:,0])
    show_img_with_mask(x[5], yp[5,:,:,0])
    show_img_with_mask(x[6], yp[6,:,:,0])
    show_img_with_mask(x[7], yp[7,:,:,0])
    show_img_with_mask(x[8], yp[8,:,:,0])

def ResultsValid(model, df):
    x_v=[]
    masks_en=[]
    name_list=os.listdir('/content/data/valid')
    path_list = list(df['img'])
    for i in range(145):
        img_path=path_list[i]
        img = np.array(Image.open(img_path))
        img = cv2.resize(img, (224, 224))
        x_v += [img]
    x_v = np.array(x_v) / 255.
    mask=model.predict(x_v)
    mask_resh=[cv2.resize(mask[i], (240,320)) for i in range(145)]
    mask_resh=[(mask_resh[i]>0.5).astype(int) for i in range(145)]
    mask_en=[encode_rle(m) for m in mask_resh]
    dic={'img': name_list, 'mask': mask_en}
    df_result=pd.DataFrame.from_dict(dic)
    return(df_result)

def get_data(df):
    x_tr=[]
    y_tr=[]
    img_list = list(df['img'])
    mask_list= list(df['mask'])
    for i in range(1315):
        img_name = img_list[i]
        mask_rle = mask_list[i]
        img = np.array(Image.open(img_name))
        mask = decode_rle(mask_rle)

        img = cv2.resize(img, (224, 224))
        mask = cv2.resize(mask, (224, 224))


        x_tr += [img]
        y_tr += [mask]

    x_tr = np.array(x_tr) / 255.
    y_tr = np.array(y_tr)
    y_tr=np.expand_dims(y_tr, -1)
    return( x_tr, y_tr)



def ResultsTest(model):
    x_v=[]
    masks_en=[]
    name_list=os.listdir('/content/data/test')
    listd=os.listdir(f"/content/data/test")
    pat=f"/content/data/test"
    im_path=sorted([(f"{pat}/{x}") for x in listd])
    for i in range(100):
        img_path=im_path[i]
        img = np.array(Image.open(img_path))
        img = cv2.resize(img, (224, 224))
        x_v += [img]
    x_v = np.array(x_v) / 255.
    mask=model.predict(x_v)
    mask_resh=[cv2.resize(mask[i], (240,320)) for i in range(100)]
    mask_resh=[(mask_resh[i]>0.5).astype(int) for i in range(100)]
    return(mask_resh)