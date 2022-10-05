import cv2 as cv
import math
import numpy as np
import sys
from camera import Camera

if (len(sys.argv) < 2):
    print(f"Usage: {sys.argv[0]} <file_to_read(.npz format)>")
    exit()


# Taken from https://github.com/opencv/opencv/blob/4.x/samples/python/plane_ar.py
ar_verts = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                       [0, 0, 0.75], [0, 1, 0.75], [1, 1, 0.75], [1, 0, 0.75],
                       [0, 0.5, 1.5], [1, 0.5, 1.5]])
ar_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
            (4, 8), (5, 8), (6, 9), (7, 9), (8, 9)]


cv.namedWindow("Webcam")
cv.namedWindow("Contours")
cv.namedWindow("AR")
cv.namedWindow("Warped")

# Capture video from webcam
camera = Camera(0, sys.argv[1])

while True:
    img = camera.get_frame()
    detected = False

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(blur)

    # edges = cv.Canny(equalized, 100, 200)
    ret, edges = cv.threshold(equalized, 200, 255, cv.THRESH_BINARY)

    contours, hierarchy = cv.findContours(
        edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=cv.contourArea, reverse=True)
    # print(hierarchy)

    approx_list = []
    for i in range(len(contours)):
        c = contours[i]

        perimeter = cv.arcLength(c, True)

        approx = cv.approxPolyDP(c, 0.04 * perimeter, True)
        # Only append if has 4 sides
        if(len(approx) == 4):
            approx_list.append(approx)
            detected = True

    img1 = img.copy()
    cv.drawContours(img, contours, -1, (0, 255, 0), 3)
    cv.drawContours(img1, approx_list, -1, (255, 0, 0), 3)

    approx_list = sorted(approx_list, key=cv.contourArea, reverse=True)

    cv.imshow("Webcam", img)
    cv.imshow("Contours", img1)

    if detected and len(approx_list[0]) == 4:
        approx_list = np.array(approx_list[0], dtype=np.float32)
        x0, y0, x1, y1 = approx_list
        quad_3d = np.float32([[1, 1, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0]])
        # approx_list = np.float32([approx_list[0][0],approx_list[0][1],approx_list[0][2],approx_list[0][3]])
        ret, rvec, tvec, _ = cv.solvePnPRansac(
            quad_3d, approx_list, camera.matrix, camera.dist_coeffs)

        if ret == True:

            # verts = ar_verts * [(x1-x0), (y1-y0), -(x1-x0)*0.3] + (x0, y0, 0)
            verts = cv.projectPoints(
                ar_verts, rvec, tvec, camera.matrix, camera.dist_coeffs)[0].reshape(-1, 2)

            img2 = img1.copy()
            for i, j in ar_edges:
                (x0, y0), (x1, y1) = verts[i], verts[j]
                if -10000 < x0 < 10000 and -10000 < x1 < 10000 and -10000 < y0 < 10000 and -10000 < y1 < 10000:
                    cv.line(img2, (int(x0), int(y0)),
                            (int(x1), int(y1)), (0, 0, 255), 2)
            dst = np.array([
                [0, 0],
                [200, 0],
                [200, 400],
                [0, 400]], dtype="float32")

            # print(approx_list)
            # sorted(approx_list, key=lambda x:x[0][0]+x[0][1], reverse=True)
            m = cv.getPerspectiveTransform(approx_list, dst)
            warp = cv.warpPerspective(img, m, (200, 400))
            cv.imshow("Warped", warp)
            cv.imshow("AR", img2)

    cv.waitKey(50)