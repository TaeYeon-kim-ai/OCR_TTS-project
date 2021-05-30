import os
import glob
import cv2

l = []

for filename in glob.glob('F:/BSW_dtaset/data2_parksunwoo/basic_rename/*.JPG'): # path to your images folder
    print(str(filename))
    n_len = len(str(filename))
    l.append(n_len)

print(l[0])
    



text = str("흔들리다 치마 두리번거리다 효도 해석")
print(len(text))