import os
import numpy as np
import natsort 
import cv2 as cv
import pandas as pd

# #train
# file_path = 'C:/final_project/IBM/image-data/hangul-images' 
# file_names = os.listdir(file_path)
# after_list = natsort.natsorted(file_names)
# print(after_list)

# #변환
# test_image_arr = []
# for i in range(1,518593):
#     path = 'C:/final_project/IBM/image-data/hangul-images/hangul_{}.jpeg'.format(i)
#     image = cv.imread(path)
#     image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     image = cv.resize(image, (64, 64), interpolation = cv.INTER_CUBIC)
#     test_image_arr.append(image)
#     if i % 1000 == 0 :
#         print(i)

# test_image_arr = np.asarray(test_image_arr)

# np.save('C:/data/npy/x_data_text.npy', arr = test_image_arr)

test_image_arr = []
for i in range(1,38):
    path = 'C:/final_project/data/test_data2/test_{}.png'.format(i)
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, (64, 64), interpolation = cv.INTER_CUBIC)
    test_image_arr.append(image)
    if i % 10 == 0 :
        print(i)

test_image_arr = np.asarray(test_image_arr)

np.save('C:/final_project/data/test_data2/test_data.npy', arr = test_image_arr)
x = np.load('C:/final_project/data/test_data2/test_data.npy')
print(x.shape)

#===============y_train_onehot=====================
########################## String Label OneHotEncoding
df = pd.read_csv('C:/final_project/IBM/image-data/labels-map.csv', header = None, encoding = 'utf-8')
y = df.iloc[:, 1].values.reshape(-1, 1)
print(y.shape)
onehot = LabelEncoder()
onehot_y_train = onehot.fit_transform(y)
print(type(onehot_y_train))
print(onehot_y_train.shape)
print(onehot_y_train)
np.save('C:/final_project/data/test_data2/onehot_y_train.npy', onehot_y_train)
