from tkinter import Scale
from camera import Camera
import cv2 as cv
import numpy as np
from model import Model
import sys

from opencv_renderer import openCVRenderer

USE_OPENGL = False

if USE_OPENGL:
    from OpenGL.GLUT import *
    from opengl_renderer import OpenGLRenderer

files = np.load("../desktop_fontao.npz")
camera_matrix = files["camera_matrix"]
distortion_coefficient = files["distortion_coefficients"]

drawing = cv.imread("../cards/full/1.png")

camera = Camera(0, sys.argv[1])
if USE_OPENGL:
    renderer = OpenGLRenderer(
        camera_matrix, int(camera.get_width()), int(camera.get_height()))
    # Models must be loaded after renderer is instantiated
    renderer.models.append(Model("../models/Cup1.obj", 0, swapyz=True,
                                 scale=[0.05, 0.05, 0.05]))
    renderer.models.append(Model("../models/Plastic_Cup.obj", 1,
                                 swapyz=True, scale=[0.03, 0.03, 0.03]))
else:
    renderer = openCVRenderer(camera_matrix, distortion_coefficient, scale=0.1)


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

    if USE_OPENGL:
        renderer.aruco_ids = ids
        renderer.tvecs = tvecs
        renderer.rvecs = rvecs
        renderer.image = frame.copy()
        glutPostRedisplay()
        glutMainLoopEvent()
    else:
        if len(ids) > 0:
            print(corners)
            src_points = [(0, 0), (drawing.shape[0]-1, 0),
                          (drawing.shape[0]-1, drawing.shape[1]-1), (0, drawing.shape[1]-1)]
            dst_points = [corners[0][0][0], corners[0][0][1], corners[0][0][2], corners[0][0][3]]

            M = cv.getPerspectiveTransform(src_points, dst_points)
            warp = cv.warpPerspective(drawing, M, (frame.shape[0], frame.shape[1]))
            
            frame = cv.fillConvexPoly(frame, dst_points, 0, 16)
            frame =  frame + warp

        renderer.display(frame, rvecs, tvecs)
        # cv.imshow("Aruco", frame)

    cv.waitKey(50)
