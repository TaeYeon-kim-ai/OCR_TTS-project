from PIL.Image import Image
import cv2
import easyocr
from gtts import gTTS
import os
import hgtk
path = "F:/final_data/test_data_OCR/ocr_test"
file = os.listdir(path)
print("list = {}".format(file))

for f in file:
    fpath = path + "/" + f
    image = cv2.imread(fpath)
    print(f) # '0001.jpg', '0002.jpg', '0003.jpg',
    # image = cv2.resize(image, dsize=(500*3, 367*3), interpolation=cv2.INTER_AREA)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    OCR_text =[] #모든 img list
    reader = easyocr.Reader(['ko']) #easyocr을 ko로 지정
    reader = reader.readtext(image) #reader로 readtext 결로상 cv2.image 하나씩 읽기
    # print(reader)
    for i in range(len(reader)): #i에 하나씩 넣어주고 출력 #len만큼 읽음
        # if reader[i][2] > 0.1:
        OCR_text.append(reader[i][1])

    OCR_text = " ".join(OCR_text)
    print(OCR_text)
    break

# a = hgtk.letter.decompose(OCR_text)
a = hgtk.text.decompose(OCR_text)
print(a)
