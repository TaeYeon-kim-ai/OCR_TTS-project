
import matplotlib.pyplot as plt
import time 
import copy
import cv2
import numpy as np
import os
import pandas as pd
from skimage.measure import compare_ssim

# 외곽선검출 train_normal
path = r'F:/final_data/test_data_OCR/ocr_test' # Source Folder
dstpath = r'C:/data/fish_data/fish_datasets/x1_train/normal' # Destination Folder

try:
    makedirs(dstpath)
except:
    print ("Directory already exist, images will be written in asme folder")
# Folder won't used
files = os.listdir(path)

for image in files:
    img = cv2.imread(os.path.join(path,image))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dst = cv2.dilate(img, k)
    gradient = cv2.morphologyEx(dst, cv2.MORPH_GRADIENT, k)
    size = cv2.resize(gradient, (256, 256), interpolation = cv2.INTER_CUBIC)
    #결과 출력
    merged = np.hstack((img, gradient))
    cv2.imshow('gradient', merged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imwrite(os.path.join(dstpath,image),size)

