import numpy as np
import pandas as pd
import cv2 as cv
from tensorflow.keras.models import load_model

Y_TRAIN_PATH = 'C:/final_project/IBM/image-data/labels-map.csv'
TEST_IMAGE_PATH = 'C:/final_project/data/test_data2/test_data.npy's
MODEL_PATH = 'C:/final_project/h5/best_model.hdf5'

# Test Image
img = cv.imread(TEST_IMAGE_PATH, cv.IMREAD_COLOR)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()
img = img.reshape(1, img.shape[0], img.shape[1], 1)
print(type(img))
print(img.shape)

# Model
model = load_model(MODEL_PATH)
model.summary()
y_pred = model.predict(img)
print(np.argmax(y_pred))
print(y_pred)

# Test
onehot_y_train = np.load('F:/Team Project/OCR/02_Image_to_Text_model/onehot_y_train.npy')
print(onehot_y_train.shape)

result = np.where(onehot_y_train == np.argmax(y_pred))
print(result)

y_train = pd.read_csv(Y_TRAIN_PATH, header = None)
y_train = y_train.iloc[:, 1]
print(y_train[result[0][0]])