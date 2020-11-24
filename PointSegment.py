import os
import cv2
import glob
import target
import numpy as np
from matplotlib import pyplot as plt


# Load config file
with open('config.yaml','r') as file:
    config = yaml.safe_load(file)


base = os.path.relpath(config['root_path'])
I = target.natsort(glob.glob(os.path.join(base,config['us_folder'],'*')))

pts =[]
for imn in I:
    im = cv2.imread(imn, 0)
    
    plt.figure('Image {}'.format(os.path.basename(imn)))
    plt.axis('off'), plt.imshow(im,cmap='gray')
    plt.show()
    p = plt.ginput(1,mouse_add=3,mouse_pop=1,mouse_stop=2,timeout=0)
    plt.close()
    
    pts.append(p[0])

# Save image points coordinates
np.save(os.path.join(base,'cross_point.npy'), np.array(pts))