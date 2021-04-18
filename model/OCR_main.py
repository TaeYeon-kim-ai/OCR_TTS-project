import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import  Dense, Flatten, Conv2D, Input, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU, Softmax, GlobalAveragePooling2D, BatchNormalization, Dropout , GaussianDropout, Activation
from tensorflow.keras.models import Sequential, Model, load_model, save_model
import pandas as pd
import os
import cv2 as cv
import natsort

# x = np.load('C:/final_project_read_your_letter/npy/x_test_128.npy', allow_pickle=True)
#CONTROLER
x_image_num = 198912
img_shape_w = 64
img_shape_h = 64
input_dim = (img_shape_w, img_shape_h, 1)
no_of_classes = 2368
bts = 32
epoch = 100

#==================================================================================

#y_label 작업
x = np.load('C:/data/npy/x_data_text.npy')
df = pd.read_csv('C:/final_project/IBM/image-data/labels-map.csv', header = None, encoding = 'utf-8')
y = df.iloc[:, 1].values.reshape(-1, 1)
# print(x.shape, y.shape)
# print(type(y))

#to_categori
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()
onehot.fit(y)
y = onehot.transform(y).toarray()
# print(x.shape, y.shape)

model = load_model("C:/final_project/h5/model1_kfold_2021.04.12_m.h5")
model.load_weights('C:/final_project/h5/best_model.hdf5')

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle = True, random_state=0)
a = 1
for train_index, val_index in kfold.split(x, y) :
    print("TRAIN:", train_index.shape, "TEST:", val_index.shape)
    x_train, x_val = x[train_index], x[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
# #EVAL
loss, acc = model.evaluate(x_val, y_val)
print("loss : ", loss)
print("acc : ", acc)
x_pred = np.load('C:/final_project/data/test_data2/test_data.npy')
x_pred = x_pred.reshape(-1, 64,64,1)
# print(x_pred)
result = model.predict(x_pred)
print(result.shape)

result = np.where(onehot_y_train == np.argmax(result))
print(result)


# key에 정수값 value에 한글
# label text에서 for문 돌려서 1~2368까지의 key value값을 만들고 
# argmax한 값을 key값으로 사용해서 value값 찾기