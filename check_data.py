'''
清洗数据集：将2号标签全变为背景255
'''

import cv2 as cv
import os
import numpy as np

root_from = "./YB/LB/"
root_save = "./YB/NewLB/"


for a,b,c in os.walk(root_from):
    for file_i in c:
        file_i_full_path = os.path.join(root_from, file_i)
        img_i = cv.imread(file_i_full_path)
        # print("img_i = ",img_i)
        img_new_i = np.where(img_i == 2, 255, img_i)
        # print("img_new_i = ",img_new_i)

        new_file_i_full_path = os.path.join(root_save, file_i)
        cv.imwrite(new_file_i_full_path,img_new_i)

# import cv2
# img_path = "YB/LB/HS_14.TIF"
#
#
# img = cv2.imread(img_path)
#
# print(img)
# print(img.shape)

