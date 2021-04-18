import os
import numpy as np

#train_nomal
file_path = 'C:/final_project/IBM/image-data/hangul-images' 
file_names = os.listdir(file_path)
file_names

i = 0 #1부터 변경할 것임
for name in file_names: 
    src = os.path.join(file_path, name)
    dst = str(i) + '.jpeg'
    dst = os.path.join(file_path, dst)
    
    os.rename(src, dst)
    i += 1