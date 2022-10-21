from camera import Camera
import cv2 as cv
import numpy as np
from model import Model
import sys

from opencv_renderer import OpenCVRenderer

USE_OPENGL = False

if USE_OPENGL:
    from OpenGL.GLUT import *
    from opengl_renderer import OpenGLRenderer

files = np.load("../params_1.npz")
camera_matrix = files["camera_matrix"]
distortion_coefficient = files["distortion_coefficients"]


camera = Camera(1, sys.argv[1])
if USE_OPENGL:
    renderer = OpenGLRenderer(
        camera_matrix, int(camera.get_width()), int(camera.get_height()))
    # Models must be loaded after renderer is instantiated
    renderer.models.append(Model("../models/Cup1.obj", 0, swapyz=True,
                                 scale=[0.1, 0.1, 0.1]))
    renderer.models.append(Model("../models/Plastic_Cup.obj", 1,
                                 swapyz=True, scale=[0.03, 0.03, 0.03]))
else:
    renderer = OpenCVRenderer(camera_matrix, distortion_coefficient, scale=0.1)


while True:

    frame = camera.get_frame()

    rvecs = []
    tvecs = []

    arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
    arucoParams = cv.aruco.DetectorParameters_create()
    corners, ids, rejected = cv.aruco.detectMarkers(frame, arucoDict,
                                                    parameters=arucoParams)
    if len(corners) == 0:
        ids = []

    if len(corners) > 0:
        cv.aruco.drawDetectedMarkers(frame, corners, ids)

        rvecs, tvecs, objPoints = cv.aruco.estimatePoseSingleMarkers(
            corners, 0.1, camera_matrix, distortion_coefficient)

        # for i in range(len(rvecs)):
        #     img = cv.drawFrameAxes(
        #         img, camera_matrix, distortion_coefficient, rvecs[i], tvecs[i], 0.05)


    renderer.aruco_ids = ids
    renderer.tvecs = tvecs
    renderer.rvecs = rvecs
    renderer.image = frame.copy()
    renderer.display()

    cv.waitKey(50)