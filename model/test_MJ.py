# import os
import pandas as pd
import cv2
import numpy as np
from pandas.core.tools.datetimes import DatetimeScalar
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPool2D, GlobalAveragePooling2D, Flatten, Dense, Input, BatchNormalization, GlobalMaxPool2D
from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.backend import conv2d
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
path = "D:\python\pjt_odo\labels-map.csv"
load = pd.read_csv(path,encoding="utf-8")

def Recognition_model():
    input = Input(shape=(64,64,3))
    x = Conv2D(filters=(64), kernel_size=(64,21),strides=1)(input)
    x = BatchNormalization()(x)
    x1 = GlobalMaxPool2D()(x)

    x = Conv2D(filters=(64), kernel_size=(21,64),strides=1)(input)
    x = BatchNormalization()(x)
    x2 = GlobalMaxPool2D()(x)

    x = Conv2D(filters=(64), kernel_size=(21,21),strides=1)(input)
    x = BatchNormalization()(x)
    x3 = GlobalMaxPool2D()(x)

    x = Conv2D(filters=(64), kernel_size=(5,5),strides=1)(input)
    x = BatchNormalization()(x)
    x4 = GlobalMaxPool2D()(x)

    # x5 = Flatten()(x1+x2+x3+x4)

    x = Dense(1512)(x5)
    output = Dense(2350, activation="softmax")(x)

    model = Model(inputs=input, outputs= output)

    model.summary()
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['acc'])
    return model


model = Recognition_model()
load = shuffle(load)

for i in range(0,16):
    step = 42300

    lst_image_path = list(load.iloc[step*i:step*(i+1),0])
    lst_image_lable = list(load.iloc[step*i:step*(i+1),-1])

    x_images = []
    for i, image_path in enumerate(lst_image_path) :
        img = cv2.imread(image_path)
        x_images.append(img)
        print(i)
    x_images = np.array(x_images)
    y_label = np.array(lst_image_lable)

    print(y_label.shape)
    print(x_images.shape)
    print(np.unique(y_label))
    # (37599,)
    # (37599, 64, 64, 3)

    y_label = y_label.reshape(-1,1)
    onthot = OneHotEncoder()
    onthot.fit(y_label)
    y_label = onthot.transform(y_label).toarray()

    print(y_label.shape)

    x_images = x_images/255.
    print(x_images.shape)

    kfold = KFold(n_splits=5, shuffle=True)
    a = 1
    for train_index, test_index in kfold.split(x_images):
        print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
        x_train, x_val = x_images[train_index], x_images[test_index]
        y_train, y_val = y_label[train_index], y_label[test_index]

        model.fit(x=x_train, y=y_train, batch_size=2350, epochs=100, validation_data=(x_val, y_val), shuffle=True)
        if a == 5:
            model.save("model1_{}.h5".format(i))
        a += 1