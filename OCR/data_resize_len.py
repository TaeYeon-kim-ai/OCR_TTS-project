import cv2
import glob

for filename in glob.glob('F:/BSW_dtaset/data2_parksunwoo/skew_resize/*.JPG'): # path to your images folder
    print(filename)
    img = cv2.imread(filename) 
    rl = cv2.resize(img, (320,64))
    cv2.imwrite(f'{filename}resized.JPG', rl)