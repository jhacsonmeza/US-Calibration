import os
import cv2
import yaml
import glob
import target
import numpy as np
from matplotlib import pyplot as plt


# Load config file
with open('config.yaml','r') as file:
    config = yaml.safe_load(file)


base = os.path.relpath(config['root_path'])
paths = [os.path.join(base, x) for x in os.listdir(base) if 'data' in x]

fig, ax = plt.subplots()
imfig = ax.imshow(np.zeros([config['h'],config['w'],3],np.uint8))
for path in paths:
    I = target.natsort(glob.glob(os.path.join(path,config['us_folder'],'*')))

    pts =[]
    for imn in I:
        im = cv2.imread(imn)
        
        imfig.set_data(im)
        ax.set_title('Image {}, from folder {}'.format(os.path.basename(imn), os.path.basename(path)))
        ax.axis('off')
        plt.pause(0.2)
        p = plt.ginput(1,mouse_add=3,mouse_pop=1,mouse_stop=2,timeout=0)
        
        pts.append(p[0])
    
    print('Dataset {} finished.'.format(os.path.basename(path)))

    # Save image points coordinates
    np.save(os.path.join(path,'cross_point.npy'), np.array(pts))