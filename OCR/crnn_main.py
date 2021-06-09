from crnn_01 import CTCLayer
import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#DATA
data_dir = Path("F:/CRNN_dataset_kr/hangul-images/")

#get list of all the images
images = sorted(list(map(str, list(data_dir.glob("*.png")))))
labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
characters = set(char for label in labels for char in label)


print("num_image : ", len(images))
print("num_label : ", len(labels))
print("num_unique chracters", len(characters))
print("chracters : ", characters)


# num_image :  22000
# num_label :  22000
# num_unique chracters 966

# batch size for train val
batch_size = 1

# image demention
img_width = 320
img_height = 56

downsample_factor = 4

max_length = max([len(label) for label in labels])


#====================================================

#preprocessing
char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=list(characters), num_oov_indices=0, mask_token=None
)

#mapping intergers back to original characters
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token = None, invert = True
    )


def split_data(images, labels, train_size = 0.9, shuffle=True) :
    size = len(images)
    indices = np.arange(size)
    
    if shuffle :
        np.random.shuffle(indices)
        # 3. get the size of training samples
    train_samples = int(size * train_size)
    
    #4, split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_val, y_val = images[indices[train_samples:]], labels[indices[train_samples:]] 
    print(x_train.shape, y_train.shape) #(6196,) (6196,)

    return x_train, x_val, y_train, y_val


#Splitting data into training and validation sets 
x_train, x_val, y_train, y_val = split_data(np.array(images), np.array(labels))
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
#(6196,) (689,) (6196,) (689,)

def encode_single_sample(img_path, label) :
    img = tf.io.read_file(img_path) # 1. 이미지 읽어오기
    img = tf.io.decode_png(img, channels=1) # 2. 디코딩, gray sacle 변환
    img = tf.image.convert_image_dtype(img, tf.float32) # 3. 실수변환 //Convert to float32 in [0, 1]range 
    img = tf.image.resize(img, [img_height, img_width]) # 4. Resize to the desired size 320, 64
    # 5. transpose the image because we want the time
    # dim to correspend to the width of the image 
    img = tf.transpose(img, perm = [1, 0, 2])
    label = char_to_num(tf.strings.unicode_split(label, input_encoding = "UTF-8")) # 6. 라벨링 문자를 숫자로 맵핑
    return {"image" : img , "label" : label} # 7. 입력 차원 두개 return

# dataset 개체만들기 
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
validation_dataset = (
    validation_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
#(6196,) (689,) (6196,) (689,)

#데이터 시각화 =======================================
# _, ax = plt.subplot(4, 4, fig_size=(10, 5))
# for batch in train_dataset.take(1):
#     images = batch["image"]
#     labels = batch["label"]
#     for i in range(16) :
#         img = (images[i]*255).numpy().astype("uint8")
#         label = tf.string.reduce_join(num_to_char(labels[i])).numpy().decode("utf-8")
#         ax[i // 4, i % 4].imshow(img[:, :, 0].T, camp="gray")
#         ax[i // 4, i % 4].set_title(label)
#         ax[i // 4, i % 4].axis("off")
    
# plt.show()

#======================================================

# CTC Model
'''
CTC이해필요
'''
class CTCLay(layers.Layer) :
    def __init__(self, name = None) :
        super().__init__(name = name)
        self.loss_fn = keras.backend.ctc_batch_cost #()

    def call(self, y_true, y_pred) :
        batch_len = tf.cast(tf.shape(y_true)[0], dtype = 'int64')
        input_length = tf.cast(tf.shape(y_pred)[1], dtype = 'int64')
        label_length = tf.cast(tf.shape(y_true)[1], dtype = 'int64')

        input_length = input_length * tf.ones(shape = (batch_len, 1), dtype='int64')
        label_length = label_length * tf.ones(shape = (batch_len, 1), dtype='int64')

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # 테스트 시간 내 계산된것 반환
        return y_pred


#======================================

#main Model
def build_model() :
    # Input to the model
    input_img = layers.Input(shape = (img_width, img_height, 1), name = "image", dtype="float32")
    labels = layers.Input(name = "label", shape = (None,), dtype='float32')

    # 1 conv block
    x = layers.Conv2D(32, 3, activation='relu', kernel_initializer="he_normal", padding="same", name = "Conv1")(input_img)
    x = layers.MaxPooling2D((2,2), name = "pool1")(x)

    # 2 conv block
    x = layers.Conv2D(64, 3, activation='relu', kernel_initializer = "he_normal", padding="same", name = "Conv2")(x)
    x = layers.MaxPooling2D((2,2), name = "pool2")(x)

    
    # 우리는 풀 크기와 스트라이드 2가있는 2 개의 최대 풀을 사용했습니다.
    # 따라서 다운 샘플링 된 기능 맵은 4 배 더 작습니다. 개수
    # 마지막 레이어의 필터는 64 개입니다.
    # 모델의 RNN 부분에 출력 전달
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape = new_shape, name = "reshape")(x)
    x = layers.Dense(64, activation='relu', name= "dense1")(x)
    x = layers.Dropout(0.2)(x)

    #RNN
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout = 0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout = 0.25))(x)

    # output layer
    x = layers.Dense(len(characters) + 1, activation='softmax', name = "dense2")(x)

    # Add CTC layers
    output = CTCLayer(name = "CTC_loss")(labels, x)

    #define the model
    model = keras.models.Model(inputs = [input_img, labels], outputs = output, name = "OCR_model_v1")

    # Optimeizer
    opt = keras.optimizers.Adam()
    
    # Complie
    model.compile(optimizer = opt)
    
    return model

# # Get the model
model = build_model()
model.summary()

#=========================================================
model.load_weights('F:/CRNN_dataset_kr/MC/best_crnn_01.hdf5')
#=========================================================
#Inference
#예측모델
prediction_model = keras.models.Model(model.get_layer(name = "image").input, model.get_layer(name = "dense2").output)
prediction_model.summary()

#model.save()

# 디코딩 네트워크의 출력을 디코딩하는 유틸리티 함수
def decode_batch_predictions(pred) :
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    # 결괍 나복 및 텍스트 가져오기
    output_text = []
    for res in results :
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("UTF-8")
        output_text.append(res)

    return output_text

model.predict(x_val, y_val)



#  Let's check results on some validation samples
# for batch in validation_dataset.take(1):
#     batch_images = batch["image"]
#     batch_labels = batch["label"]

#     preds = prediction_model.predict(batch_images)
#     pred_texts = decode_batch_predictions(preds)

#     orig_texts = []
#     for label in batch_labels:
#         label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("UTF-8")
#         orig_texts.append(label)


#     _, ax = plt.subplots(4, 4, figsize=(15, 5))
#     for i in range(len(pred_texts)):
#         img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
#         img = img.T
#         title = f"Prediction: {pred_texts[i]}"
#         ax[i // 4, i % 4].imshow(img, cmap="gray")
#         ax[i // 4, i % 4].set_title(title)
#         ax[i // 4, i % 4].axis("off")
        
# plt.show()
