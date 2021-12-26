import numpy as np
import cv2 
import matplotlib.pyplot as plt
import os



base_path = '/home/mkz/catkin_ws/src/road_seg_deeplabv3/data/masks/2k'
save_path = '/home/mkz/catkin_ws/src/road_seg_deeplabv3/data/masks/cropped'

mono = True

files = os.listdir(base_path)

for file in files:
    if mono:
        img = cv2.imread(os.path.join(base_path,file),cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(os.path.join(base_path,file))
    
    if mono:
        img_c = img[:,256:768]
    else:
        img_c = img[:,256:768,:]
    
    cv2.imwrite(os.path.join(save_path,file),img_c)
