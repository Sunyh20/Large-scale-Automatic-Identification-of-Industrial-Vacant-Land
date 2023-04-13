import cv2
import random
import numpy as np
import os
import argparse
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
# 预测函数规整
import joblib
from tensorflow.keras.models import load_model
from osgeo import gdal
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img,dtype="float") / 255.0
    return img


def read_tif2(path):
    dataset = gdal.Open(path)
    # 栅格矩阵的列数
    im_width = dataset.RasterXSize
    # 栅格矩阵的行数
    im_height = dataset.RasterYSize
    # 波段数
    im_bands = dataset.RasterCount
    im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
    im_proj = dataset.GetProjection()  # 获取投影信息
    li = []
    for i in range(1, im_bands + 1):
        band = dataset.GetRasterBand(i)
        im_datas = band.ReadAsArray(0, 0, im_width, im_height)
        li.append(im_datas)
    return np.array(li), im_width, im_height, im_geotrans, im_proj
    # 释放内存。如果不释放，在arcgis或envi中打开该图像时显示文件已被占用
    del dataset


def writeTiff(im_data, im_width, im_height, im_bands, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
        # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

    
def predict():
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model('seg_fc_hrnet_avgpooling-model.h5')
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        # #load the image
        print(path)
        image = load_img(img_path+"\\"+path)
        image = [image]
        image = np.array(image)
        qq, im_width, im_height, im_geotrans, im_proj = read_tif2(img_path+"\\"+path)
        out = model.predict(image)
        out = out[0,:,:]
        li = []
        for i in range(out.shape[0]):
            if np.argmax(out[i,:]) == 0:
                res = 0
            if np.argmax(out[i, :]) == 1:
                res = 1
            if np.argmax(out[i, :]) == 2:
                res = 2
            li.append(res)
        arr_li = np.array(li).reshape(im_width,im_height)
        writeTiff(arr_li, im_width, im_height, 1, im_geotrans, im_proj, out_path+"\\"+path.split(".")[0]+'_pre.TIF')

if __name__ == '__main__':

    img_path = r"YCDT_2015"
    out_path = r'YC_2015'
    dir = os.path.exists(out_path)
    if dir != True:
        os.makedirs(out_path)
    TEST_SET = [files for root, dirs, files in os.walk(img_path)][0]
    # TEST_SET = ['1234_CJ155.TIF', '1234_CJ160.TIF']
    image_size = 256
    predict()



