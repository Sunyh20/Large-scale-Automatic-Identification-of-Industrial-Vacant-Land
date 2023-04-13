#coding=utf-8
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder  
from PIL import Image  
import matplotlib.pyplot as plt  
import cv2
import random
import os
from tqdm import tqdm
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import pickle

import tensorflow as tf
#
# gpus = tf.config.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(device=gpu, enable=True)
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 7  
np.random.seed(seed)  
  
#data_shape = 360*480  
img_w = 256
img_h = 256
#有一个为背景  
# n_label = 4+1
#
# classes = [0. ,  1.,  2.,   3.  , 4.]

n_label = 3

def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)/255.0
    return img


filepath ='./YB/'

def get_train_val(val_rate = 0.25):
    train_url = []    
    train_set = []
    val_set  = []
    for pic in os.listdir(filepath + 'YX'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i]) 
        else:
            train_set.append(train_url[i])
    return train_set,val_set

# data for training  
def generateData(batch_size,data=[]):  
    #print 'generateData...'
    while True:  
        train_data = []  
        train_label = []  
        batch = 0  
        for i in (range(len(data))): 
            url = data[i]
            batch += 1 
            img = load_img(filepath + 'YX/' + url)
            train_data.append(img)  
            label = load_img(filepath + 'LB/' + url, grayscale=True)
            label = label.reshape(img_w * img_h)
            label2 = []
            for da in label:
                if da == 255:
                    da2 = [1, 0, 0]
                if da == 1:
                    da2 = [0, 1, 0]
                if da == 2:
                    da2 = [0, 0, 1]
                label2.append(da2)
            # print label.shape  
            train_label.append(label2)
            if batch % batch_size==0: 
                #print 'get enough bacth!\n'
                train_data = np.array(train_data)  
                train_label = np.array(train_label)
                # print(train_data.shape,train_label.shape)
                yield (train_data,train_label)  
                train_data = []  
                train_label = []  
                batch = 0  
 
# data for validation 
def generateValidData(batch_size,data=[]):  
    #print 'generateValidData...'
    while True:  
        valid_data = []  
        valid_label = []  
        batch = 0  
        for i in (range(len(data))):  
            url = data[i]
            batch += 1  
            img = load_img(filepath + 'YX/' + url)
            valid_data.append(img)  
            label = load_img(filepath + 'LB/' + url, grayscale=True)
            label = label.reshape(img_w * img_h)
            label2 = []
            for da in label:
                if da == 255:
                    da2 = [1, 0, 0]
                if da == 1:
                    da2 = [0, 1, 0]
                if da == 2:
                    da2 = [0, 0, 1]
                label2.append(da2)
            # print label.shape  
            valid_label.append(label2)
            if batch % batch_size==0:  
                valid_data = np.array(valid_data)  
                valid_label = np.array(valid_label)
                # print(valid_data.shape,valid_label.shape)
                yield (valid_data,valid_label)
                valid_data = []  
                valid_label = []  
                batch = 0

def SegNet():
    model = Sequential()
    #encoder
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(img_w,img_h,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    #(128,128)
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2),data_format="channels_last"))
    #(64,64)
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2),data_format="channels_last"))
    #(32,32)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2),data_format="channels_last"))
    #(16,16)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2),data_format="channels_last"))
    #(8,8)
    #decoder
    model.add(UpSampling2D(size=(2,2)))
    #(16,16)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))
    #(32,32)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))
    #(64,64)
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))
    #(128,128)
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))
    #(256,256)
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_w, img_h,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(n_label, (1, 1), strides=(1, 1), padding='same'))
    model.add(Reshape((img_w*img_h,n_label)))
    #axis=1和axis=2互换位置，等同于np.swapaxes(layer,1,2)
    # model.add(Permute((2,1)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

def train():
    EPOCHS = 50
    BS = 5
    # model = SegNet()

    # from seg_fc_hrnet import seg_fc_hrnet
    # model = seg_fc_hrnet()
    # model.summary()
    from seg_fc_hrnet_avgpooling import seg_fc_hrnet
    model = seg_fc_hrnet()
    model.summary()

    train_set, val_set = get_train_val()
    train_numb = len(train_set)
    valid_numb = len(val_set)
    print ("the number of train data is",train_numb)
    print ("the number of val data is",valid_numb)
    train_data = generateData(BS,train_set)
    valid_data = generateValidData(BS,val_set)
    H = model.fit_generator(generator=train_data,steps_per_epoch=train_numb/BS,epochs=EPOCHS,verbose=1,
                    validation_data=valid_data,validation_steps=valid_numb/BS)

    # 保存H
    with open('./trainHistoryDict_50_test.data', 'wb') as file_pi:
        pickle.dump(H.history, file_pi)

    #保存模型方法1
    model_save_path = r"seg_fc_hrnet_avgpooling-model_50_test.h5"
    model.save(model_save_path)


    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on SegNet Satellite Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")

    plt.show()
    plt.savefig("result_test.png")


if __name__=='__main__':
    print("正在训练模型...")
    train()
