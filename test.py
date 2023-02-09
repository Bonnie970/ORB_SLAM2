from lib.ORB_SLAM2 import orb_extractor_wrapper, test_size
import numpy as np
import ctypes
import cv2

import sys,os
sys.path.append('/ORB_SLAM2')

imgs = ['img.jpeg', 'img2.jpeg']

for f in imgs: 
    img = cv2.imread(f) # np.eye(50)
    print(test_size(img))


    k = np.zeros((2000, 2), dtype=np.float32)
    d = np.zeros((2000, 256), dtype=np.uint8)
    (rows, cols) = (img.shape[0], img.shape[1])
    # orb_extractor_wrapper(np.ctypeslib.as_ctypes(img).data, rows, cols, k, d)
    # orb_extractor_wrapper(img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), rows, cols, k, d)
    r = orb_extractor_wrapper(np.uint8(img), rows, cols, k, d)

    print(k,'\n###',d,'\n###', r)

print('passed test')
