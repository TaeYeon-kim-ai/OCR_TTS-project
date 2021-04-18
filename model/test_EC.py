import os
import pandas as pd
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image

##### CONSTANTS #####
# Default paths.
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  './labels/2350-common-hangul.txt')
DEFAULT_DATA_DIR = os.path.join(SCRIPT_PATH, 'image-data')
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_PATH, 'saved-model')

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

DEFAULT_NUM_EPOCHS = 100
BATCH_SIZE = 32

# This will be determined by the number of entries in the given label file.
num_image = 169200
num_classes = 2350
keep_prob = 0.5


##### Prepare train data #####
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

print(hangul_number)

# 판다스로 csv파일을 읽어옴
df = pd.read_csv(os.path.join(DEFAULT_DATA_DIR, 'labels-map.csv'), header=None) # [9400 rows x 2 columns]

train_images = np.empty((num_image, IMAGE_WIDTH, IMAGE_HEIGHT, 1))
train_labels = np.empty((num_image), dtype=int)

# df에 있는 각 경로에 대해서 반복하여
for idx, img_path in enumerate(df.iloc[:, 0]):
    img = image.load_img(img_path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), color_mode="grayscale")
    img = image.img_to_array(img).astype('float32')/255.
    train_images[idx,:,:,:] = img

    ganada = df.iloc[idx,1]
    train_labels[idx] = hangul_number[ganada]

from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
print((train_labels.shape)) # (9400, 2350)

print("[info] train_images, train_labels 완료")











##############################################

# preprocessing (one hot encoding, because "softmax")

# Create the model!
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
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

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
er = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
re = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.3, verbose=1)

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['acc'])
model.fit(x=train_images, y=train_labels, batch_size=BATCH_SIZE, epochs=DEFAULT_NUM_EPOCHS, callbacks=[er], validation_split=0.2)

# loss = model.evaluate()

if __name__ == '__main__':
    model.save(os.path.join(DEFAULT_MODEL_DIR, 'mymodel.h5'))
