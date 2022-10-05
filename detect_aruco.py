import cv2 as cv
import numpy as np

ar_verts = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                       [0, 0, 0.75], [0, 1, 0.75], [1, 1, 0.75], [1, 0, 0.75],
                       [0, 0.5, 1.5], [1, 0.5, 1.5]])
ar_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
            (4, 8), (5, 8), (6, 9), (7, 9), (8, 9)]

# ar_verts = np.float32([[0, 0, 0], [0, 1, 0]])
# ar_edges = [(0, 1)]

ar_verts = ar_verts * 0.05


files = np.load("desktop_fontao.npz")
camera_matrix = files["camera_matrix"]
distortion_coefficient = files["distortion_coefficients"]

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Can't open stream")
    exit()

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

        verts = cv.projectPoints(
                ar_verts, rvecs[0], tvecs[0], camera_matrix, distortion_coefficient)[0].reshape(-1, 2)
        for i, j in ar_edges:
            (x0, y0), (x1, y1) = verts[i], verts[j]
            if -10000 < x0 < 10000 and -10000 < x1 < 10000 and -10000 < y0 < 10000 and -10000 < y1 < 10000:
                cv.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
                
    cv.imshow("Aruco", img)
    cv.waitKey(50)
