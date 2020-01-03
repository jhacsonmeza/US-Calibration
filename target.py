import re
import cv2
import itertools
import numpy as np

def natsort(l):
    '''
    Lambda function for nautural sorting of strings. Useful for sorting the 
    list of file name of images with the target. Taken from:
    https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/
    
    input:
        l: list of input images with the target
    output:
        Nutural sorted list of images
    '''
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    
    return sorted(l, key=alphanum_key)


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


def detect(im, global_th=True, th_im=False):
    '''
    Function for detection of 3 concentric circle targets
    
    input:
        im: image where targets will be detected
        global_th: True if binarize image usign global thresholding with the
                    Otsu method. False if use (local) adaptive thresholding
        th_im: True if return thresholdized image with bounding boxes,
                False if not
        
    output:
        * image with drawn bounding boxes
        * 3 x 1 x 2 matrix with image coordinates of each target
        * True if detection succeeds and False if fail
    '''
    
    # Convert im to gray, binarize with adaptive threshold or global Otsu
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    if global_th:
        _,bw = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,61,25)
    
    
    # Create structuring element, apply morphological opening operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    imo = cv2.morphologyEx(bw,cv2.MORPH_OPEN,kernel)
    
    # Compute contours
    contours,_ = cv2.findContours(imo,cv2.RETR_TREE,
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
        if (len(approx) > 5) & (area > 30) & (area < 40000):
            conts.append(cnt)
            areas.append(area)
            perimeters.append(perimeter)
            
            M = cv2.moments(cnt)
            c.append([M['m10']/M['m00'],M['m01']/M['m00']])
    
    # Convert lists to numpy arrays
    c = np.array(c)
    conts = np.array(conts)
    areas = np.array(areas)
    perimeters = np.array(perimeters)
    
    
    
    # Traget is composed of three circles. Each circle in turn is composed of
    # two concentric circles (a black and a white circle) which have the sames
    # cooridnate center, and the distance between they should be zero
    dist = np.array([])
    ind = np.array([], np.int)
    for i in range(len(c)-1):
        d = np.linalg.norm(c[i]-c[i+1]) # Distance between adjacent contours
        
        # If distance is smaller than a threshold, save value and index
        if d < 2:
            dist = np.append(dist, d)
            ind = np.append(ind, i)
    
    
    # Find our circles if there are more than three contours
    ret = True
    if len(dist) > 3: # In this case we have the circles and other contours
        # Area and perimeter of the contours of our three circles should have
        # appriximately the same values. We select the three smaller values
        # of subtracting the median from both area and perimter
        score = abs(np.median(areas[ind]) - areas[ind])/np.median(areas[ind]) \
        + abs(np.median(perimeters[ind]) - perimeters[ind])/np.median(
                perimeters[ind])
        
        ind = ind[np.argsort(score)[:3]] # Indices of smaller three values
        
    
    # Obtain the contours and center coordinates of the circles
    circ = conts[ind]
    c = c[ind]
    
        
    # Draw bounding boxes in the detections
    if th_im:
        im = np.dstack([imo, imo, imo])
        
    for cnt in circ:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)

    
    return ret, im, c.reshape(-1,1,2)


def match(c1, c2, F):
    '''
    Find the corresponding point in image 2 of the centers in the firsrt view
    by estimating the epipolar constraint (x'^T F x = 0) between the points in
    the first view and all the possible correspondences in the second view.
    Matches with lowest epipolar constraint scores (ideally 0) are the true
    correspondences.
    
    input:
        c1: 3 x 1 x 2 matrix with image coordinates of concentric circle
            centers in camera 1
        c2: 3 x 1 x 2 matrix with image coordinates of concentric circle 
            centers in camera 2
        F: 3 x 3 Fundamental matrix
        
    output:
        * c2 correspondences of points c1 (c2 rearranged).
    '''
    # Convert points to homogeneous
    x1 = cv2.convertPointsToHomogeneous(c1)
    x2 = cv2.convertPointsToHomogeneous(c2)
    
    # Permutation of all possible mathces of the three points in the 2nd view
    indexes = np.array(list(itertools.permutations([0,1,2])))
    
    # Estimate the epipolar constraint with all the possible matches of c1
    # epc^ijk = x'^ijr F_rs x^ks
    epc = np.einsum('ijr,rs,ks->ijk', x2[indexes].reshape(6,3,3), F, x1[:,0,:])
    # Scores are in the diagonal of each 2D array of the 3D tensor
    # We look for the lowest norm of the values in the diagonal
    ind = np.argmin(np.linalg.norm(np.einsum('ijj->ij',epc),axis=1))
    
    return c2[indexes[ind]]


def centers3D(P1, P2, c1, c2):
    '''
    Function to compute the 3D coordinates of centers of each concentric circle
    through triangulation. c1 and c2 have the unarranged (unmatched) image 
    coordinates of centers, hence based on reprojection error we look for 
    correct correspondences, i.e., the pair of points with minimum reprojection
    error are considered correspondences.
    
    input:
        P1: projection matrix of camera 1
        P2: projection matrix of camera 2
        c1: 3 x 1 x 2 matrix with image coordinates of concentric circle 
            centers in camera 1
        c2: 3 x 1 x 2 matrix with image coordinates of concentric circle 
            centers in camera 2
        
    output:
        C matrix with unlabel 3D coordinates of centers
    '''
    
    C = []
    for p in c1:
        d = np.array([])
        pts3D = []
        for p1, p2 in itertools.product([p],c2):
            # Triangulate
            X = cv2.triangulatePoints(P1,P2,p1.T,p2.T)
            
            # Reproject Points
            x = P1 @ X
            x = x[:2]/x[-1]
            
            # Save point and convert from homogeneous to Euclidean
            pts3D.append(X[:3]/X[-1])
            
            # Reprojection error in pixels
            d = np.r_[d, np.linalg.norm(x.flatten()-p1)]
        
        ind = np.argsort(d)[0]
        C.append(np.array(pts3D)[ind])
        c2 = np.delete(c2, ind, 0)
    
    return np.hstack(C) # Each column is a 3D point


def label(X):
    '''
    Function to label the 3D coordinates of the centers of each concentric 
    circle
    
    input:
        X: 3 (dims) x 3 (points num.) matrix with 3D coordinates of centers
        
    output:
        * Xo 3D coordinates of point in the target that represent the origin
        * Xx 3D coordinates of point in the target in direction of x-axis
        * Xy 3D coordinates of point in the target in direction of y-axis
    '''
    
    # Label: find 3D point in origin and 3D points in x and y directions using
    # distance and perpendicularity criteria
    a = 0
    for X1, X2, X3 in itertools.permutations(X.T,3):
        # We assume X1 is the origin and X2 and X3 points in x and y directions
        # (in any order). Then, X2 is closer to X1.
        if np.linalg.norm(X2-X1) > np.linalg.norm(X3-X1):
            continue
        
        # vec(X1,X2) and vec(X1,X3) are perpendicular and dot product must be
        # zero or small, we looking for the set with the smaller dot product
        dot = np.dot(X2-X1, X3-X1)
        
        if a == 0:
            min_dot = dot
            Xo = X1
            Xx = X2
            Xy = X3
            a = 1
            
            continue
        
        if min_dot > dot:
            min_dot = dot
            Xo = X1
            Xx = X2
            Xy = X3
    
    return Xo, Xx, Xy


def drawAxes(im1, K1, dist1, R, t):
    '''
    Function to project and draw the axes of the target coordinate system in 
    the image plane 1
    
    input:
        im1: image of camera 1
        K1: intrinsic matrix of camera 1
        dist1: distortion coefficients
        R, t: position and orientation of target frame relative to camera 1
        
    output:
        image with axes drawn
    '''
    axes = 40*np.array([[0,0,0], [1.,0,0], [0,1.,0], [0,0,1.]]) # axes to draw
    
    # Reproject target coordinate system axes
    rvec, _ = cv2.Rodrigues(R)
    axs, _ = cv2.projectPoints(axes, rvec, t, K1, dist1)
    axs = np.int32(axs[:,0,:])
    
    # Draw axes
    origin = tuple(axs[0])
    im1 = cv2.line(im1, origin, tuple(axs[1]), (0,0,255), 5)
    im1 = cv2.line(im1, origin, tuple(axs[2]), (0,255,0), 5)
    im1 = cv2.line(im1, origin, tuple(axs[3]), (255,0,0), 5)
    
    return im1


def drawCub(im1, K1, dist1, R, t):
    '''
    Function to project and draw a cube from the target coordinate system in 
    the image plane 1
    
    input:
        im1: image of camera 1
        K1: intrinsic matrix of camera 1
        dist1: distortion coefficients
        R, t: position and orientation of target frame relative to camera 1
        
    output:
        image with cube drawn
    '''
    vertices = 40*np.array([[0,0,0],[0,1.,0],[1,1,0],[1,0,0],[0,0,1],
                            [0,1,1],[1,1,1],[1,0,1]]) # vertices to draw
    
    # Reproject vertices from target coordinate system
    rvec, _ = cv2.Rodrigues(R)
    vert, _ = cv2.projectPoints(vertices, rvec, t, K1, dist1)
    vert = np.int32(vert[:,0,:])
    
    # draw ground floor in green
    im1 = cv2.drawContours(im1, [vert[:4]], -1, (0,255,0), -3)
    
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        im1 = cv2.line(im1, tuple(vert[i]), tuple(vert[j]), (255), 3)
    
    # draw top layer in red color
    im1 = cv2.drawContours(im1, [vert[4:]], -1, (0,0,255), 3)
    
    return im1


def drawCenters(im1, im2, K1, K2, R, t, dist1, dist2, Xo, Xx, Xy):
    '''
    Function to draw the reprojected 3D centers in both images.
    Red point Xo, green point Xx, blue point Xy.
    
    input:
        im1, im2: left and right images
        K1, K2: intrinsic matrices of camera 1 and camera 2
        R, t: intrinsic parameters between camera 1 and camera 2
        dist1, dist2: distortion coefficients
        Xo, Xx, Xy: 3D coordinates of the center of circles
    
    output:
        images with drawn reprojected centers
    '''
    X = np.c_[Xo,Xx,Xy].T
    rvec, _ = cv2.Rodrigues(R)
    
    c1, _ = cv2.projectPoints(X, np.zeros((3,1)), np.zeros((3,1)), K1, dist1)
    c2, _ = cv2.projectPoints(X, rvec, t, K2, dist2)
    
    im1 = cv2.circle(im1,(int(c1[0,0,0]),int(c1[0,0,1])),8,(0,0,255),-1)
    im1 = cv2.circle(im1,(int(c1[1,0,0]),int(c1[1,0,1])),8,(0,255,0),-1)
    im1 = cv2.circle(im1,(int(c1[2,0,0]),int(c1[2,0,1])),8,(255,0,0),-1)
    
    im2 = cv2.circle(im2,(int(c2[0,0,0]),int(c2[0,0,1])),8,(0,0,255),-1)
    im2 = cv2.circle(im2,(int(c2[1,0,0]),int(c2[1,0,1])),8,(0,255,0),-1)
    im2 = cv2.circle(im2,(int(c2[2,0,0]),int(c2[2,0,1])),8,(255,0,0),-1)
    
    return im1, im2


def drawEpilines(img, x, view, F):
    '''
    Function to draw three epipolar lines in the image.
    
    input:
        img: image on which draw the epilines
        x: 3 x 1 x 2 array with points in a first view to estimate epipolar
            lines in a second view
        view: index of the image (1 or 2) that contains the points
        F: Fundamental matrix
    
    output:
        image with epilines
    '''
    lines = cv2.computeCorrespondEpilines(x, view, F)
    
    r,c,_ = img.shape
    color = ((255,0,0),(0,255,0),(0,0,255))
    for i,r in enumerate(lines):
        x0,y0 = map(int, [0, -r[0,2]/r[0,1] ])
        x1,y1 = map(int, [c, -(r[0,2]+r[0,0]*c)/r[0,1] ])
        img = cv2.line(img, (x0,y0), (x1,y1), color[i],5)
    return img


def getPose(Xo, Xx, Xy):
    '''
    Function to compute pose of target
    
    input:
        Xo: 3D coordinates of point in the target that represent the origin
        Xx: 3D coordinates of point in the target in direction of x-axis
        Xy: 3D coordinates of point in the target in direction of y-axis
        
    output:
        R: rotation matrix
        t: translation vector
    '''
    
    xaxis = Xx-Xo # Vector pointing to x direction
    xaxis = xaxis/np.linalg.norm(xaxis) # Conversion to unitary
    
    yaxis = Xy-Xo # Vector pointing to y direction
    yaxis = yaxis/np.linalg.norm(yaxis) # Conversion to unitary
    
    zaxis = np.cross(xaxis, yaxis) # Unitary vector pointing to z direction
    
    # Build rotation matrix and translation vector
    R = np.c_[xaxis,yaxis,zaxis]
    t = Xo
    
    return R, t