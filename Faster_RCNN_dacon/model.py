import random
import pprint
import sys
import time
import numpy as np
import pickle
import math
import cv2
import copy
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os

from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout, Add
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Progbar
from tensorflow.keras.layers import Layer
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

#참고 : https://dacon.io/competitions/open/235560/codeshare/613?page=1&dtype=recent
#Write train.csv to annotation.txt
#train.csv 파일 안의 내용을 txt 파일로 변경합니다. 이 txt파일은 나중에 데이터를 파싱하는 과정에서 사용됩니다.
#txt 파일은 filename, x1, x2, y1, y2, classname을 ,로 구분하여 저장합니다. 여기서 x1, x2, y1, y2는 바운딩 박스의 좌표입니다.

base_path = os.getcwd() #현재 작업경로
print(base_path) #C:\final_project
train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
# For training
f= open(base_path + "/annotation_train.txt","w+")
for idx, row in train_df.iterrows():
    sys.stdout.write('Parse train_imgs ' + str(idx) + '; Number of boxes: ' + str(len(train_df)) + '\r')
    sys.stdout.flush()
    x1 = int(row['bbox_x1'])
    x2 = int(row['bbox_x2'])
    y1 = int(row['bbox_y1'])
    y2 = int(row['bbox_y2'])
    className = row['class']
    fileName = os.path.join(base_path, 'train', row['img_file'])

    f.write(str(fileName) + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + str(className) + '\n')
f.close()


#Builde pretrained model
# 사전 훈련된 모델 적재
# ResNet50사전 훈련모델 사용
#activation_39층, feature map으로 사용, model정의 후 가중치 저장
# RPN과 classification layer에서 사용

from tensorflow.keras.applications.resnet50 import ResNet50

resnet50 = ResNet50(weight = 'imagenet', include_top=False)
output = resnet50.get_layer('activation_39').output

model = Model(resnet50.input, output)

#resnet50.save_weights(os.path.join(base_path, 'model/', 'resnet50.h5')) # save pre-trained model to load weights later

#RPN layer