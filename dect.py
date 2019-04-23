import os
import cv2
import glob
import itertools
import numpy as np
import scipy.io as sio

def dst(l,p):
    '''
    Function to compute the 2D distance between a point and a line
    
    inputs:
        l: a, b and c coefficients of a line given by ax + by + c = 0
        p: (x, y, 1) point in homogeneous coordinates
    
    output:
        distance
    '''
    if p.ndim == 1:
        # if a single point is given like np.array([x,y,1])
        # convert to np.array([[x,y,1]])
        p = np.array([p])
      
    return abs(l[0]*p[:,0]+l[1]*p[:,1]+l[2])/np.sqrt(l[0]**2+l[1]**2)


def detection(im,n):
    '''
    Function for detection of n concentric circle targets
    
    input:
        im: image where targets will be detected
        n: number of concentric circles to detect
        
    output:
        * image with drawn bounding boxes
        * n x 2 matrix with image coordinates of each target
    '''
    
    # Convert im to gray, binarize with Otsu's method
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#    _,bw = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY,21,2)
    
    # Create structuring element, apply morphological opening operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    imo = cv2.morphologyEx(bw,cv2.MORPH_OPEN,kernel)
    
    # Compute contours and then areas and perimeters of it.
    _,contours,_ = cv2.findContours(imo,cv2.RETR_TREE,\
                                        cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(c) for c in contours])
    perimeters = np.array([cv2.arcLength(c,1) for c in contours])
    
    # Evaluate circularity criteria and take contours that meet it 
    # R = 4*pi*a/p^2. For a circle R = 1
    R = 4*np.pi*areas/perimeters**2
    circ = np.array(contours)[(R > 0.85*R.max()) & (areas > 1000)]
      
    
    # Compute centroids of each contour in circ
    c = []
    for cont in circ:
        M = cv2.moments(cont)
        c.append([M['m10']/M['m00'],M['m01']/M['m00']])
    c = np.array(c)
      
    
    '''# As targets are concentric circles, both circles have the same center, 
    # and distance between these centers should be zero
    d = np.array([])
    for i in range(len(c)-1):
        d = np.append(d,np.linalg.norm(c[i]-c[i+1]))
      
    # Take the first 3 contours with smaller distances 
    ind = np.argsort(d)[:n]
    c = c[ind]
    ring = circ[ind]'''
    ring = circ.copy()
    
    # Draw bounding boxes in the detections
    #im = np.dstack([imo, imo, imo])
    for cnt in ring:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
    
#    return im, c, np.array([cv2.contourArea(c) for c in ring])
    return im, c
    

def label(im,c):
    '''
    Function to label the different cocenctric circles as 0, x and y, where 0
    represent the origin of the coordinate system, x the direction in the x
    axis and y the direction to the y axis.
    
    input:
        im: image with three concentric circles to label
        c: 3 x 2 matrix with coordinates of concentric circle centers
        
    output:
        * image with 0, x and y text drawn in each respective target
        * origin coordinate
        * x label coordinate
        * y label coordinate
    '''
    
    # Permutation with all possible combination of orders in which matrix
    #  centers can come
    a = 0
    for pt1, pt2, pt3 in itertools.permutations(c,3):
        # We assume pt1 is the origin and pt2 and pt3 the x and y axis (in any 
        # order) so vec(pt1,pt2) and vec(pt1,pt3) are perpendicular and dot
        # product must be zero or small, we looking for the set 
        # with the smaller dot product
        dot = np.dot(pt2-pt1,pt3-pt1)
        if a == 0:
                dot_min = dot
                org = pt1
                p1 = pt2
                p2 = pt3
                
                a = 1
                continue
        if dot_min > dot:
                dot_min = dot
                org = pt1
                p1 = pt2
                p2 = pt3
    
    # Now we are sure that org is the right origin coordinate, let's 
    # figure out which is the correct x and y points using a criteria of 
    # distance (x is closer to org than y)
    
    # Line between org and p1 and distance to p2
    l1 = np.cross(np.append(org,1),np.append(p1,1))
    d1 = dst(l1,p2)[0]
    
    # Line between org and p2 and distance to p1
    l2 = np.cross(np.append(org,1),np.append(p2,1))
    d2 = dst(l2,p1)[0]
    
    # Define x and y
    if d1 < d2:
        x = p2
        y = p1
    else:
        x = p1
        y = p2
    
    # If org is below y, check that x is to the right of org
    if (y[1] < org[1]) & (org[0] > x[0]):
        temp = org
        org = x
        x = temp
    # If org is obove y, check that x is to the left of org
    elif (y[1] > org[1]) & (org[0] < x[0]):
        temp = org
        org = x
        x = temp
    
    # Write the correct labels on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im,'0',tuple(np.int32(org)), font, 1,(0,0,255),3)
    cv2.putText(im,'x',tuple(np.int32(x)), font, 1,(0,0,255),3)
    cv2.putText(im,'y',tuple(np.int32(y)), font, 1,(0,0,255),3)
    
    return im, org, x, y


def drawAxes(img, origin, imgpts):
    '''
    Function to draw the axes of a coordinate system
    
    input:
        img: image to draw cube
        origin: origin coordinates of target frame
        imgpts: image coordinates of the ends of axes
        
    output:
        image with axes drawn
    '''
    
    corner = tuple(np.int32(origin))
    img = cv2.line(img,corner,tuple(np.int32(imgpts[0].ravel())), (255,0,0),5)
    img = cv2.line(img,corner,tuple(np.int32(imgpts[1].ravel())), (0,255,0),5)
    img = cv2.line(img,corner,tuple(np.int32(imgpts[2].ravel())), (0,0,255),5)
    return img


def drawCub(img, imgpts):
    '''
    Function to draw a cube in an image
    
    input:
        img: image to draw cube
        imgpts: image coordinates of the vertices of the cube. First lower
                vertices coordinates and then upper vertices.
        
    output:
        image with cube drawn
    '''
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img


cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detection', 640*2, 512)

I1 = sorted(glob.glob('acquisitionUS/L/*.jpg'), key=os.path.getmtime)
I2 = sorted(glob.glob('acquisitionUS/R/*.jpg'), key=os.path.getmtime)

Params = sio.loadmat('Params.mat')
K1 = Params['K1']
K2 = Params['K2']
R = Params['R']
t = Params['t']

P1 = K1 @ np.hstack([np.eye(3),np.zeros([3,1])])
P2 = K2 @ np.hstack([R,t])

axes = 40*np.array([[1.,0,0], [0,1.,0], [0,0,1.]]) # axes for drawAxes
#axes = 20*np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
#                   [0,0,3],[0,3,3],[3,3,3],[3,0,3]]) # axes for drawCub

pts3D = []
pim1 = []
pim2 = []
for im1n, im2n in zip(I1,I2):
    im1 = cv2.imread(im1n)
    im2 = cv2.imread(im2n)
    
    im1, c1 = detection(im1,3)
    im2, c2 = detection(im2,3)
    
    if len(c1) != 3:
        print('Circles in image {} couldn\'t be detected'.format(
                im1n.split('\\')[-1]))
        continue
    elif len(c2) != 3:
        print('Circles in image {} couldn\'t be detected'.format(
                im2n.split('\\')[-1]))
        continue
    
    im1, org1, x1, y1 = label(im1,c1)
    im2, org2, x2, y2 = label(im2,c2)
    pim1.append(np.r_[org1,x1,y1].tolist())
    pim2.append(np.r_[org2,x2,y2].tolist())
    
    p1 = np.array([org1,x1,y1]).T
    p2 = np.array([org2,x2,y2]).T
    
    X = cv2.triangulatePoints(P1,P2,p1,p2)
    X = X[:3]/X[-1]
    pts3D.append(X.T.flatten().tolist())
    
    xaxis = X[:,1]-X[:,0]
    xaxis = xaxis/np.linalg.norm(xaxis)
    yaxis = X[:,2]-X[:,0]
    yaxis = yaxis/np.linalg.norm(yaxis)
    zaxis = np.cross(xaxis, yaxis)
    
    Rm = np.array([xaxis,yaxis,zaxis]).T
    rvec, _ = cv2.Rodrigues(Rm)
    tvec = X[:,0].reshape(3,-1)
    ax, _ = cv2.projectPoints(axes, rvec, tvec, K1, None)
    
    img = drawAxes(im1.copy(), org1, ax)
#    img = drawCub(im1.copy(), ax)
    
    cv2.imshow('Detection',np.hstack([img,im2]))
    if cv2.waitKey(100) & 0xFF == 27:
        break

cv2.destroyAllWindows()
pts3D = np.array(pts3D).T
#sio.savemat('pts3D.mat',{'X':pts3D})
#sio.savemat('pts2D.mat', {'pim1':np.array(pim1), 'pim2': np.array(pim2)})