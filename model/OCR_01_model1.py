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
x_image_num = 15344
img_shape_w = 64
img_shape_h = 64
input_dim = (img_shape_w, img_shape_h, 1)
no_of_classes = 274
bts = 32
epoch = 100

#==================================================================================

#y_label 작업
path = os.path.dirname(os.path.abspath(__file__)) 
label_file = os.path.join(path, 'C:/final_project/IBM/labels/274-common-hangul.txt')


hangul_label = {}
common_hangul = open(label_file, "r", encoding='utf-8')
i = 0
while True :
    hangul = common_hangul.readline().strip()
    hangul_label[str(hangul)] = i
    i += 1
    if hangul == "":
        break
common_hangul.close()
print("한글 대 숫자 대응 딕셔너리 완성")
print(hangul_label)


#라벨링할 csv파일
df = pd.read_csv('C:/final_project/IBM/image-data/labels-map.csv', header = None, encoding = 'utf-8')
train_images = np.empty((x_image_num, img_shape_w, img_shape_h, 1))
train_labels = np.empty((x_image_num), dtype=int)


#df에 있는 각 경로 반복하여 csv에 라벨링
from tensorflow.keras.preprocessing import image
for idx, img_path in enumerate(df.iloc[:, 0]): #enumerrate 리스트가 있는ㅇ 경우 순서와 리스트의 값을 전달 idx행 숫자 별, csv파일의 img_path전달 0번째 열에 루트있음
    img = image.load_img(img_path, target_size=(img_shape_w, img_shape_h), color_mode="grayscale") #경로에 있는 이미지를 지정된 size와 컬러로 변환
    img = image.img_to_array(img).astype('float32')/255. #불러온 이미지 정규화
    train_images[idx,:,:,:] = img #train_image에 넣기 출력값 = image수, w, h, color

    ganada = df.iloc[idx,1] #ganada에는 label해줄 한글 단어 들어감 df경로 내 1번째 컬럼에 있음
    train_labels[idx] = hangul_label[ganada] #가나다가 들어간 한글 라벨을 train_idx에 삽입


#to_categori
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
print(train_labels.shape)

print("[info] train_images, train_labels 완료")

#===========================================================================

x = train_images
y = train_labels
print(x.shape, y.shape) #(21504, 64, 64, 1) (21504, 256)

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

    x = Dense(2048)(x)
    output = Dense(no_of_classes, activation = 'softmax')(x)

    model = Model(inputs = inputs, outputs = output)
    model.summary()

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
    return model

model = create_model()

#COMPILE   

from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
mc = ModelCheckpoint('C:/final_project/h5/best_model_{epoch:02d}-{val_loss:.4f}.hdf5', save_best_only=True, mode = 'auto')
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
    if a == 5:
        model.save("model1_{}.h5".format(i))
    a += 1

    # #EVAL
    # loss, acc = model.evaluate(x_val, y_val)
    # print("loss : ", loss)
    # print("acc : ", acc)