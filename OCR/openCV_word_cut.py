import cv2
import numpy as np
from functools import cmp_to_key


#좌표 정렬 함수
def xy_compare(x, y):
    if x[1] > y[1]: # y좌표가 작은 것부터 앞으로
        return 1
    elif x[1] == y[1]: # y좌표가 같을 경우
        if x[0] > y[0]: # x 좌표가 작은 것이 앞으로 나오게
            return 1
        elif x[0] < y[0]: # x 좌표가 큰 것이 뒤로
            return -1
        else: # 같은 경우에는 그대로
            return 0
    else: # y좌표가 큰 것이 뒤로
        return -1


img = cv2.imread('F:/CRNN_dataset_kr/test_data/test_img/korexp5.png')
rgb = cv2.pyrDown(img)

#추출용 이미지 따로만들기
make_word_image = cv2.pyrDown(img)

small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 3))
grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

# using RETR_EXTERNAL instead of RETR_CCOMP
contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
mask = np.zeros(bw.shape, dtype=np.uint8)

annotations = []

for idx in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[idx])
    #annotation 추출
    annotations.append([x, y, w, h])

    #bbox그리기
    mask[y:y+h, x:x+w] = 0
    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
    if r > 0.45 and w > 8 and h > 8:
        cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
        
# bbox 1차 확인
cv2.imshow('rects', rgb)
cv2.waitKey()


#bbox_image 추출작업
output_filepath = "F:/CRNN_dataset_kr/openCV_predict_img/"

#좌표 list부여
x_1 = []
y_1 = []
w_1 = []
h_1 = []

for i in range(len(annotations)) :
    print(i)
    x_1.append(annotations[i][0])
    y_1.append(annotations[i][1])
    w_1.append(annotations[i][2])
    h_1.append(annotations[i][3])

#list확인
print(x_1)
print(y_1)
print(w_1)
print(h_1)
name_rule = list(range(len(annotations)))
for i, j in zip(reversed(range(len(annotations))), name_rule) :
    #원본 img 좌표 그리기
    crop_img = make_word_image[y_1[i]:y_1[i] + h_1[i], x_1[i]:x_1[i] + w_1[i]]
    print(crop_img)
    cv2.imshow("cropped", crop_img)
    cv2.waitKey(0)
    cv2.imwrite(output_filepath + 'word_crop_{}.jpg'.format(j), crop_img)
