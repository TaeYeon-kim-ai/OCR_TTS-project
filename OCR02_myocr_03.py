import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import imghdr
'''
Morphologicla Transformation은 이미지를 Segmentation하여 단순화, 
제거, 보정을 통해서 형태를 파악하는 목적으로 사용이 됩니다. 
일반적으로 binary나 grayscale image에 사용이 됩니다. 사용하는 방법으로는 Dilation(팽창), 
Erosion(침식), 그리고 2개를 조합한 Opening과 Closing이 있습니다. 여기에는 2가지 Input값이 있는데, 
하나는 원본 이미지이고 또 다른 하나는 structuring element입니다.
'''

#cv2.getStructuringElement
# RECT = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) #사각형 검출
# ELLIPSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) #타원형 검출
# CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)) #십자가로 검출

#cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
# src 입력 // dst 출력 // MorphType op 시행할 연산 *Close=닫기연산 // element 구조체 // iteration 반복횟수, 안적을 시 default 1
# ※ 닫기연산 은 팽창연산 - 침식연산을 순서대로 해준것과 같음. 흰색 오브젝트 내의 검은점을 삭제하는데 사용

# cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
# cv2.drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]]) → dst
# Parameters:	image – 원본 이미지 // contours – contours정보. // contourIdx – contours list type에서 몇번째 contours line을 그릴 것인지. -1 이면 전체 // color – contours line color
# thickness – contours line의 두께. 음수이면 contours line의 내부를 채움.

path = "F:/final_data/test_OCR_1"
file = os.listdir(path)

for img in file : 
    large = cv2.imread(os.path.join(path, img))
    rgb = cv2.pyrDown(large)
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) #그레이 변환
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 7)) #구조체 정의 ()
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #이미지 이진화
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # connected = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    #connected = cv2.morphologyEx(bw, cv2.MORPH_GRADIENT, kernel)
    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(bw.shape, dtype=np.uint8)


    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx]) #contour에 외적하고 있는 직사각형 얻기
        mask[y:y+h, x:x+w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

        #word = []
        if r > 0.45 and w > 7 and h > 7:
            cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 1)
            #word.append(cv2.boundingRect(rgb))

    # 시각
    plt.subplot (121), plt.imshow (large), plt.title ( 'Original' )
    plt.xticks ([]), plt.yticks ([])
    plt.subplot (122), plt.imshow (rgb), plt.title ( 'Blurred' )
    plt.xticks ([]), plt.yticks ([])
    plt.show ()












