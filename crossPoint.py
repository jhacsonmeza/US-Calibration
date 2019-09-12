# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:10:26 2019

@author: Jhacson Meza
"""

import os
import cv2
import glob
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt


base = os.path.relpath('Calibration datasets/Calibration test 19-09-12/data2')


I = sorted(glob.glob(os.path.join(base,'UScrop','*.bmp')),key=os.path.getctime)

pts =[]
T_P_W = []
for imn in I:
    im = cv2.imread(imn, 0)
    
    plt.figure('Image {}'.format(os.path.basename(imn)))
    plt.axis('off'), plt.imshow(im,cmap='gray')
    plt.show()
    p = plt.ginput(1,mouse_add=3,mouse_pop=1,mouse_stop=2,timeout=0)
    plt.close()
    
    pts.append(p[0])

pts = np.array(pts)

sio.savemat(os.path.join(base,'crossP.mat'), {'crossP':pts})