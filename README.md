# Freehand Ultrasound Calibration

Calibration of a 3D freehand ultrasound system composed of a stereo vision system of two conventional cameras, and a marker of three circles attached to the ultrasound probe. The code for the acquisition is available [here](https://github.com/jhacsonmeza/StereoBaslerUltrasound), where the stereo vision cameras and the ultrasound images are acquired simultaneously.

The calibration process can be carried out with the following three snippets of code:
* `ProbePose.py`: a snippet of code for pose estimation of the target attached to the ultrasound probe. It is needed the stereo calibration parameters and the images of both cameras. As output, we have a file `target_pose.npy` with the nx4x4 transformation of the n frames from the Target coordinate system to the (left camera) World system.

* `PointSegment.py`: manual segmentation of the cross point in the ultrasound images. You can select the point with the right-click. As input, you need the ultrasound images, and the output is a file `cross_point.npy` with an nx2 array with all the n selected center locations.

* `UScalib.py`: the calibration process is carried out here. For this, we need more than one dataset. With more than one dataset we can evaluate the reproducibility and assure a good quality of the calibration. As input we need the size of the ultrasound image in pixel, in addition to the `target_pose.npy` and `cross_point.npy` generated with the last two snippets of code. The output is a `report.txt` with the calibration reproducibility measured in 5 points of the ultrasound image (4 corners and center). Finally, a file `USparams.npz` with x and y scales factors of the ultrasound image and the transformation matrix from the ultrasound image to the target coordinate system.

In addition to the above, `target.py` contains different functions for target detection and pose estimations. Furthermore `calibration.py` contains the `Calibration` class which handles the calibration procedure. If your calibration was done with Matlab, you can use `create_calib_params.m` to edit calibration variables to be used correctly for target pose estimation.