import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import imghdr

'''
1. 문자열 추출 - 21
2. 모음인식 및 한글유형 분류 - 22
3. 초, 종성, 분리 - 23
4. 초, 종성 인식 - 24
5. 인식결과 저장 - 25

1. 이미지 이진화
가. DoG(Difference of Gaussians) filter 적용 - 이진화 명도대비 조명영향 받은 자연이미지
나. 수평/수직 누적 히스토그램
a. 문자영역 구분/문자 결합
다. 한글 자음 모음 결합
a. Consonant, vowel detection - 자음모음 분류
b. Specification of detection area center - 자음 모음의 센터점 구하기
c. Comparison of Center Point Destance - 센터점 사이의 거리 구하기
d. Consonant vowel comvination - 거리좁혀서 결합하기
'''
# 가. DOG
path = "F:/final_data/test_OCR_1"

file = os.listdir(path)

box1 = []

for image in file:
    img1 = cv2.imread(os.path.join(path, image))
    img = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) #그레이
    low_sigma = cv2.GaussianBlur(img,(3,3),0)
    high_sigma = cv2.GaussianBlur(img,(5,5),0)
    dog = low_sigma - high_sigma
    ret, dst = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(dst, contours, -1, (0,255,0), 3)
    
    for i in range(len(contours)) :
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        rect_area = w*h #area_size
        aspect_ratio = float(w)/h

        if (aspect_ratio>=0.2)and(aspect_ratio<=1.0)and(rect_area>=100)and(rect_area<=700):
            cv2.rectangle(dst,(x,y),(x+w,y+h),(0,255,0),3)
            box1.append(cv2.boundingRect(cnt))


    plt.subplot (121), plt.imshow (img1), plt.title ( 'Original' )
    plt.xticks ([]), plt.yticks ([])
    plt.subplot (122), plt.imshow (box1), plt.title ( 'Blurred' )
    plt.xticks ([]), plt.yticks ([])
    plt.show ()