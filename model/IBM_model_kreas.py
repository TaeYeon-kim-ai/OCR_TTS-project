import os
import cv2
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


##### <CONSTANTS> #####
""" Run img_gen.py file in tools subdirectory. AND
You should have 2350-common-hangul.txt file in the specifed path below
"""
# Default paths.
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__)) # c:\Users\ai\PycharmProjects\group_proj

DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH, './labels/2350-common-hangul.txt')
DEFAULT_DATA_DIR = os.path.join(SCRIPT_PATH, 'image-data')
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_PATH, 'saved-model')
DEFAULT_NPY_DIR = os.path.join(SCRIPT_PATH, 'npy')

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

DEFAULT_NUM_EPOCHS = 100
BATCH_SIZE = 32

# This will be determined by the number of entries in the given label file.
num_image = 9400
num_classes = 2350
keep_prob = 0.5
##### </CONSTANTS> #####

def make_data(csv_path):
       # key 한글 : value 숫자 대응하는 딕셔너리
       hangul_number = {} 
       common_hangul = open(DEFAULT_LABEL_FILE, "r", encoding='utf-8')
       i = 0
       while True:
              hangul = common_hangul.readline().strip()
              hangul_number[str(hangul)] = i
              i += 1
              if hangul == "":
                     break
       common_hangul.close()
       print("[info] 한글 대 숫자 대응 딕셔너리 완성")

       # 판다스로 csv파일을 읽어옴
       df = pd.read_csv(csv_path, header=None) # [num_image rows x 2 columns]

       train_images = np.empty((num_image, IMAGE_WIDTH, IMAGE_HEIGHT, 1))
       train_labels = np.empty((num_image), dtype=int)

       # df에 있는 각 경로에 대해서 반복하여
       for idx, img_path in enumerate(df.iloc[:, 0]):
              img = image.load_img(img_path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), color_mode="grayscale")
              img = image.img_to_array(img).astype('float32')/255.
              train_images[idx,:,:,:] = img

              ganada = df.iloc[idx,1]
              train_labels[idx] = hangul_number[ganada]
       train_labels = to_categorical(train_labels) # (num_image, 2350)
       print("[info] train_images, train_labels 완료")

       return train_images, train_labels 

def make_model(train_images, train_labels):
       # Create the model!
       model = Sequential([
       # First convolutional layer. 32 feature maps.
       Conv2D(filters=32, kernel_size=5,
              strides=(1, 1), padding='same',
              activation='relu',
              input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1)),
       MaxPool2D(pool_size=(2, 2), strides=2,
              padding='same'),

       # Second convolutional layer. 64 feature maps.
       Conv2D(filters=64, kernel_size=5,
              strides=(1, 1), padding='same',
              activation='relu'),
       MaxPool2D(pool_size=(2, 2), strides=2,
              padding='same'),
       
       # Third convolutional layer. 128 feature maps.
       Conv2D(filters=128, kernel_size=3,
              strides=(1, 1), padding='same',
              activation='relu'),
       MaxPool2D(pool_size=(2, 2), strides=2,
              padding='same'),
       
       # Fully connected layer. Here we choose to have 1024 neurons in this layer.
       Flatten(),
       Dense(units=1024, activation='relu'),

       # Dropout layer. This helps fight overfitting.
       Dropout(rate=keep_prob),

       # Classification layer.
       Dense(units=num_classes, activation='softmax')
       ])

       er = EarlyStopping(monitor='val_loss', patience=15, mode='auto')
       re = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.3, verbose=1)

       model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['acc'])
       model.fit(x=train_images, y=train_labels, batch_size=BATCH_SIZE, epochs=DEFAULT_NUM_EPOCHS, callbacks=[er, re], validation_split=0.2)
       return model


if __name__ == '__main__':
       train_images, train_labels = make_data(os.path.join(DEFAULT_DATA_DIR, 'labels-map.csv'))
       np.save(os.path.join(DEFAULT_NPY_DIR, 'train_images.npy'), arr=train_images)
       np.save(os.path.join(DEFAULT_NPY_DIR, 'train_labels.npy'), arr=train_labels)
       model = make_model(train_images, train_labels)
       model.save(os.path.join(DEFAULT_MODEL_DIR, 'mymodel.h5'))

# np.save('path.npy', arr=대상)
# np.load('path.npy')
# from tensorflow.keras.models import load_model
# model = load_model('path.h5')
# loss, acc = model.evaluate(x_test, y_test)
# y_predict = model.predict(x_test)