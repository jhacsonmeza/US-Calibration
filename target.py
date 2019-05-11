import cv2
import itertools
import numpy as np

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


def detection(im, global_th=False, th_a=1500, th_im=False):
    '''
    Function for detection of 3 concentric circle targets
    
    input:
        im: image where targets will be detected
        global_th: True if binarize image usign global thresholding with the
                    Otsu method. False if use (local) adaptive thresholding
        th_a: Threshold area of detected objects used to evaluate whether 
                detection succeeds or fail
        th_im: True if return thresholdized image with bounding boxes,
                False if not
        
    output:
        * image with drawn bounding boxes
        * 3 x 2 matrix with image coordinates of each target
        * True if detection succeeds and False if fail
    '''
    
    # Convert im to gray, binarize with adaptive threshold or global Otsu
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    if global_th:
        _,bw = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,61,20)
    
    
    # Create structuring element, apply morphological opening operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    imo = cv2.morphologyEx(bw,cv2.MORPH_OPEN,kernel)
    
    # Compute contours
    _,contours,_ = cv2.findContours(imo,cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop over the contours, approximate the contour with a reduced set of 
    # points and save contour if meets certain conditions with its centroid
    # area and perimeter
    c = []
    conts = []
    areas = []
    perimeters = []
    for cnt in contours:
        # Compute perimeter and approximate contour
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01*perimeter, True)
        
        # Compute area
        area = cv2.contourArea(cnt)
        
        # Check if approximated contour is stored with its area, perimeter
        # and centroid
        if (len(approx) > 5) & (area > 30) & (area < 4000):
            conts.append(cnt)
            areas.append(area)
            perimeters.append(perimeter)
            
            M = cv2.moments(cnt)
            c.append([M['m10']/M['m00'],M['m01']/M['m00']])
    
    # Convert lists to numpy array
    c = np.array(c)
    conts = np.array(conts)
    areas = np.array(areas)
    perimeters = np.array(perimeters)
    
    
    
    # As targets are concentric circles, both circles have the same coordinate 
    # center, and distance between these centers should be zero
    d = np.array([])
    for i in range(len(c)-1):
        d = np.append(d,np.linalg.norm(c[i]-c[i+1]))
      
    # Take the first 5 contours with smaller neighboring centers distances,
    # which would be potential circles
    ind = np.argsort(d)[:5] # Index of smaller distances
    c = c[ind]
    circ = conts[ind] # Potential 5 contours to be circles
    areas = areas[ind]
    perimeters = perimeters[ind]
    
    
    # Evaluate circularity criteria. For a circle R = 1
    R = 4*np.pi*areas/perimeters**2
    
    # Adjust a circle in the contours and save the radius
    r = np.array([])
    for cnt  in circ:
        _, radius = cv2.minEnclosingCircle(cnt)
        r = np.append(r,radius)
        
    
    # To take the three circles between the five contours, area, circularity
    # and the adjusted radius in three of the five contours should have 
    # approximately the same values. 
    # Subtracting and dividing by the median in each feature measured and 
    # adding them, the three smaller values are the three circles.
    v = abs(np.median(areas) - areas)/np.median(areas) + \
    abs(np.median(R) - R)/np.median(R) + abs(np.median(r) - r)/np.median(r)
    
    # Take the three smaller elements of v
    ind = np.argsort(v)[:3]
    c = c[ind] # Update centroids
    circ = circ[ind] # Update circle contours
    
    
    # Check if detection succeeds or fail.
    # If at least one element in v is larger than 0.35, the three circles were
    # not detected. This threshold value is empirical.
    ret = False if sum(v[ind] > 0.35) else True
    
        
    # Draw bounding boxes in the detections
    if th_im:
        im = np.dstack([imo, imo, imo])
        
    for cnt in circ:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)

    
    
    return im, c, ret
    

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


def targetPose(P1, P2, p1, p2):
    '''
    Function to compute pose of target
    
    input:
        P1: Pose camera 1
        P2: Pose camera 2
        p1: 2 x n array with center coordinates of targets in image 1
        p2: 2 x n array with center coordinates of targets in image 2
        
    output:
        R: rotation matrix
        t: translation vector
    '''
    X = cv2.triangulatePoints(P1,P2,p1,p2)
    X = X[:3]/X[-1] # Convert coordinates from homogeneous to Euclidean
    
    xaxis = X[:,1]-X[:,0] # Vector pointing to x direction
    xaxis = xaxis/np.linalg.norm(xaxis) # Conversion to unitary
    
    yaxis = X[:,2]-X[:,0] # Vector pointing to y direction
    yaxis = yaxis/np.linalg.norm(yaxis) # Conversion to unitary
    
    zaxis = np.cross(xaxis, yaxis) # Unitary vector pointing to z direction
    
    
    R = np.array([xaxis,yaxis,zaxis]).T
    t = X[:,0]
    
    return R, t