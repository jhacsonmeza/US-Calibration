# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:46:11 2019

@author: Jhacson Meza
"""

import os
import cv2
import glob


base = 'Calibration test 08-05-19/'



I = sorted(glob.glob(base+'acquisitionUS/US/*.bmp'), key=os.path.getmtime)

storage_path = base+'acquisitionUS/UScrop'

if not os.path.exists(storage_path):
    os.makedirs(storage_path)

for imn in I:
    image_name = os.path.basename(imn).split('.')
    
    im = cv2.imread(imn, 0)
    imc = im[67:67+400,293:293+230]
    
    cv2.imwrite('{}/{}.{}'.format(storage_path, image_name[0], 
                image_name[1]), imc)