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
x_image_num = 518592
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
print(x.shape, y.shape) #(15343, 64, 64, 3) (15343, 2)
print(type(y))

#to_categori
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()
onehot.fit(y)
y = onehot.transform(y).toarray()
print(x.shape, y.shape)

#===========================================================================

# 레이어 1
def create_model() :
    inputs = Input(shape = input_dim)
    x = Conv2D(128, 4, padding="same", activation = 'relu')(inputs)
    x = Dropout(0.1)(x)
    x1 = MaxPooling2D(2)(x)

    x = Conv2D(128, 4, padding="same", activation = 'relu')(inputs)
    x = Dropout(0.1)(x)
    x2 = MaxPooling2D(2)(x)

    x = Conv2D(128, 4, padding="same", activation = 'relu')(inputs)
    x = Dropout(0.1)(x)
    x3 = MaxPooling2D(2)(x)

    x = Conv2D(128, 4, padding="same", activation = 'relu')(inputs)
    x = Dropout(0.1)(x)
    x4 = MaxPooling2D(2)(x)
    
    x = Conv2D(128, 4, padding="same", activation = 'relu')(inputs)
    x = Dropout(0.1)(x)
    x5 = MaxPooling2D(2)(x)

    x = Conv2D(128, 4, padding="same", activation = 'relu')(inputs)
    x = Dropout(0.1)(x)
    x6 = MaxPooling2D(2)(x)

    x7 = Flatten()(x1 + x2 + x3 + x4 + x5 + x6)

    x = Dense(2048)(x7)
    output = Dense(no_of_classes, activation = 'softmax')(x)

    model = Model(inputs = inputs, outputs = output)
    model.summary()

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
    return model

model = create_model()

#COMPILE   

from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
mc = ModelCheckpoint('C:/final_project/h5/best_model.hdf5', save_best_only=True, mode = 'auto')
es = EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto')


from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle = True, random_state=0)
a = 1
for train_index, val_index in kfold.split(x, y) :
    print("TRAIN:", train_index.shape, "TEST:", val_index.shape)
    x_train, x_val = x[train_index], x[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model.fit(x_train, y_train, 
            #steps_per_epoch = x_train[-1]/32, 
            epochs = epoch,
            verbose = 1,
            validation_data=(x_val, y_val),
            callbacks = [mc, es, rl]
            )

model.save("C:/final_project/h5/model1_kfold_2021.04.12_m.h5")
model.save_weights('C:/data/h5/model1_kfold_2021.04.12_w.h5')
    
    # #EVAL
    # loss, acc = model.evaluate(x_val, y_val)
    # print("loss : ", loss)
    # print("acc : ", acc)

