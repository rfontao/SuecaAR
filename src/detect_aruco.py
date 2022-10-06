import cv2 as cv
import numpy as np
from OpenGL.GLUT import *
from opengl_renderer import OpenGLRenderer


files = np.load("../desktop_fontao.npz")
camera_matrix = files["camera_matrix"]
distortion_coefficient = files["distortion_coefficients"]

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Can't open stream")
    exit()

renderer = OpenGLRenderer(camera_matrix, 640, 480)
renderer.load_model("../models/capsule.obj")

while True:

    ret, img = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        exit()

    arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
    arucoParams = cv.aruco.DetectorParameters_create()
    corners, ids, rejected = cv.aruco.detectMarkers(img, arucoDict,
                                                    parameters=arucoParams)

    if len(corners) > 0:
        cv.aruco.drawDetectedMarkers(img, corners, ids)

        rvecs, tvecs, objPoints = cv.aruco.estimatePoseSingleMarkers(
            corners, 0.1, camera_matrix, distortion_coefficient)

        # for i in range(len(rvecs)):
        #     img = cv.drawFrameAxes(
        #         img, camera_matrix, distortion_coefficient, rvecs[i], tvecs[i], 0.05)

        renderer.rvec = rvecs[0]
        renderer.tvec = tvecs[0]
        renderer.image = img.copy()

        glutPostRedisplay()
        glutMainLoopEvent()

    cv.imshow("Aruco", img)
    cv.waitKey(50)
