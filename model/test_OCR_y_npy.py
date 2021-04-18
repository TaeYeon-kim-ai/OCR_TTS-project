import numpy as np
import pandas as pd
import cv2 as cv
import tensorflow as tf
import cv2 as cv
import glob
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization, Input, Flatten, GlobalMaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet101


# Data가 String일때 to_categorical 하는법
# 1. pandas.get_dummies(y_train)
#
# 2. from sklearn.preprocessing import LabelEncoder: String Data -> Number Data Change
#    code = np.array(code)
#    label_encoder = LabelEncoder()
#    vec = label_encoder.fit_transform(code)
#
# 3. integer_mapping = {x: i for i,x in enumerate(code)}
#    vec = [integer_mapping[word] for word in code]

'''
######################### 1. Check Image Data Length
img_path = glob.glob('F:/Team Project/OCR/02_Image_to_Text_model/image-data/my_hangul_images/hangul-images/*.jpeg')
print(len(img_path))
# 67140ea Image Data

######################### 2. Read Image Data
first_path = 'F:/Team Project/OCR/02_Image_to_Text_model/image-data/my_hangul_images/hangul-images/hangul_'
last_path = '.jpeg'
img_list = []
for i in range(1, len(img_path) + 1):
    img = cv.imread(first_path + str(i) + last_path, cv.IMREAD_COLOR)
    img_list.append(img)
x_train = np.array(img_list)
print(x_train.shape)

######################### 3. Save x_train image data
np.save('F:/Team Project/OCR/02_Image_to_Text_model/image-data/my_hangul_images/ocr_x_train_color.npy', x_train)
'''

########################## Path
X_TRAIN_PATH = 'F:/Team Project/OCR/02_Image_to_Text_model/image-data/my_hangul_images/ocr_x_train_color.npy'
Y_TRAIN_PATH = 'F:/Team Project/OCR/02_Image_to_Text_model/image-data/my_hangul_images/labels-map.csv'
MODEL_SAVE_PATH = 'F:/Team Project/OCR/02_Image_to_Text_model/model_checkpoint/04_image_to_text_inwoo.hdf5'

########################## Read Input Data
x_train = np.load(X_TRAIN_PATH)
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
# print(x_train.shape)

########################## Read Label Data
y_train = pd.read_csv(Y_TRAIN_PATH, header = None)
y_train = y_train.iloc[:, 1]

########################## String Label OneHotEncoding
onehot = LabelEncoder()
onehot_y_train = onehot.fit_transform(y_train)
# print(type(onehot_y_train))
# print(onehot_y_train.shape)
# print(onehot_y_train)
# np.save('F:/Team Project/OCR/02_Image_to_Text_model/onehot_y_train.npy', onehot_y_train)

# result = np.where(onehot_y_train == 18)
# print(result)
# print(result[0][0])
# print(y_train[result[0][0]])

y_categorical = to_categorical(onehot_y_train.reshape(-1, 1))
print(y_categorical.shape)


# train_test_split
x_train = x_train/255.
x_train, x_val, y_train, y_val = train_test_split(x_train, y_categorical, train_size = 0.8, shuffle = True)

# print('x_train: ', x_train.shape)
# print('y_train: ', y_train.shape)
# print('x_val: ', x_val.shape)
# print('y_val: ', y_val.shape)

########################## Model
def mnist_model(drop = 0.2, shape = 64, channel = 1, lr_rate = 0.001):
    input = Input(shape=(64,64,3))
    
    x = Conv2D(256, kernel_size = (3, 3), strides = (1, 1))

    output = Dense(2238, activation="softmax")(x)

    model = Model(inputs=input, outputs= output)

    model.compile(optimizer = Adam(learning_rate = lr_rate), loss = 'categorical_crossentropy', metrics = ['acc'])
    return model
model = mnist_model()
model.summary

# Model Compile and Fit
es = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'auto')
rl = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, patience = 5)
cp = ModelCheckpoint(filepath = MODEL_SAVE_PATH, monitor = 'val_loss', save_best_only = True, mode = 'auto')
# hist = model.compile(optimizer = Adam(learning_rate = 0.001),
#                      loss = 'categorical_crossentropy',
#                      metrics = ['acc'])
model.fit(x_train, y_train
          , batch_size = 32
          , epochs = 100
          , validation_data = (x_val, y_val)
          , callbacks = [es, rl, cp])