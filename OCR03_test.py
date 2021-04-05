import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img_original = cv.imread("F:/final_data/test_OCR_1/test_ocr.jpg", cv.IMREAD_GRAYSCALE)

# gamma correction
lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
# calculating the new values
    lookUpTable[0, i] = np.clip(pow(i / 255.0, 2) * 255.0, 0, 255)
    # mapping the new values with the original
gamma_corrected_img = cv.LUT(img_original, lookUpTable)

# DOG
blur1 = cv.GaussianBlur(gamma_corrected_img, (3, 3), 1, borderType=cv.BORDER_REPLICATE)
blur2 = cv.GaussianBlur(gamma_corrected_img, (7, 7), 2, borderType=cv.BORDER_REPLICATE)

dog_img = cv.subtract(blur1, blur2)

plt.imshow(dog_img)
# contrast equalisation
# step 1
alpha = 0.1
tau = 10
temp1 = pow(np.abs(dog_img), alpha)
meanImg = np.mean(temp1)

Contrast_Equa_step01 = dog_img / pow(meanImg, 1/alpha)
# step 2
minMat = np.abs(Contrast_Equa_step01)
minMat[minMat > tau] = tau
temp2 = pow(minMat, alpha)
meanImg2 = np.mean(temp2)
Contrast_Equa_step02 = Contrast_Equa_step01 / pow(meanImg2, 1/alpha)
CEqualized_img = tau * np.tanh((Contrast_Equa_step02/tau))

# plt.imshow(G.originalImage)
# plt.title('OriginalImage', fontsize=15)
# plt.xticks([]), plt.yticks([])