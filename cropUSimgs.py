# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:46:11 2019

@author: Jhacson Meza
"""

import os
import cv2
import glob


base = os.path.relpath('Calibration datasets/Calibration test 19-09-12/data2')



I = sorted(glob.glob(os.path.join(base,'US','*.bmp')), key=os.path.getmtime)

storage_path = os.path.join(base,'UScrop')

if not os.path.exists(storage_path):
    os.makedirs(storage_path)

for imn in I:
    image_name = os.path.basename(imn).split('.')
    
    im = cv2.imread(imn, 0)
#    imc = im[67:67+400,293:293+230] # 7cm depth
    imc = im[67:67+408,247:247+321] # 5cm depth
    
    cv2.imwrite('{}/{}.{}'.format(storage_path, image_name[0], 
                image_name[1]), imc)