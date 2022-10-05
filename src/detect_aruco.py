import cv2 as cv
import numpy as np

from moderngl_test import HeadlessTest
from pyrr import Matrix44

from gl import draw_scene, initOpengl, load_model




files = np.load("../desktop_fontao.npz")
camera_matrix = files["camera_matrix"]
distortion_coefficient = files["distortion_coefficients"]

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Can't open stream")
    exit()

ht = HeadlessTest()
# ht.run()
P = ht.intrinsic2Project(camera_matrix, 640, 480)

print(P)

while True:
    ret, img = cap.read()
    detected = False

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

        T = ht.extrinsic2ModelView(rvecs[0], tvecs[0])

        # ht.transform = P*T
        # gl = ht.render()
        # gl = np.array(gl) 
        # cv.imshow("gl", gl)
        # gl = cv.cvtColor(gl, cv.COLOR_RGBA2BGR)

        # dst = cv.addWeighted(img, 0.5, gl, 0.5, 0)
        # cv.imshow("Result", dst)

        
        # verts = cv.projectPoints(
        #         ar_verts, rvecs[0], tvecs[0], camera_matrix, distortion_coefficient)[0].reshape(-1, 2)
        # for i, j in ar_edges:
        #     (x0, y0), (x1, y1) = verts[i], verts[j]
        #     if -10000 < x0 < 10000 and -10000 < x1 < 10000 and -10000 < y0 < 10000 and -10000 < y1 < 10000:
        #         cv.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)

        initOpengl(640, 480)
        load_model()
        image = draw_scene(img, P, T)




    cv.imshow("Aruco", img)
    cv.waitKey(50)
