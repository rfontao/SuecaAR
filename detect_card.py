import cv2 as cv
import math
import numpy as np

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

files = np.load("desktop_fontao.npz")
camera_matrix = files["camera_matrix"]
distortion_coefficient = files["distortion_coefficients"]


t1 = 100
t2 = 200

# Capture video from webcam
cap = cv.VideoCapture(2)
if not cap.isOpened():
    print("Can't open stream")
    exit()

while True:
    ret, img = cap.read()
    detected = False

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        exit()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(blur)

    edges = cv.Canny(equalized, t1, t2)

    contours, hierarchy = cv.findContours(
        edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=cv.contourArea, reverse=True)
    # print(hierarchy)

    filtered_contours = []
    approx_list = []
    for i in range(len(contours)):
        c = contours[i]

        # Check if contour is inside another contour
        if hierarchy[0, i, 3] != -1:
            continue

        peri = cv.arcLength(c, True)
        contourArea = cv.contourArea(c)

        # Check if contour is closed
        if contourArea <= cv.arcLength(c, True):
            continue

        approx = cv.approxPolyDP(c, 0.04 * peri, True)
        # Only append if has 4 sides
        if(len(approx) == 4):
            filtered_contours.append(c)
            approx_list.append(approx)
            detected = True

    img1 = img.copy()
    cv.drawContours(img, contours, -1, (0, 255, 0), 3)
    cv.drawContours(img1, approx_list, -1, (255, 0, 0), 3)

    approx_list = sorted(approx_list, key=cv.contourArea, reverse=True)


    cv.imshow("Webcam", img)
    cv.imshow("Contours", img1)

    if detected and len(approx_list[0]) == 4:
        approx_list = np.array(approx_list[0],dtype=np.float32)
        x0, y0, x1, y1 = approx_list
        quad_3d = np.float32([[1, 1, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0]])
        # approx_list = np.float32([approx_list[0][0],approx_list[0][1],approx_list[0][2],approx_list[0][3]])
        ret, rvec, tvec, _ = cv.solvePnPRansac(
            quad_3d, approx_list, camera_matrix, distortion_coefficient)

        if ret == True:
            
            # verts = ar_verts * [(x1-x0), (y1-y0), -(x1-x0)*0.3] + (x0, y0, 0)
            verts = cv.projectPoints(
                ar_verts, rvec, tvec, camera_matrix, distortion_coefficient)[0].reshape(-1, 2)

            img2 = img1.copy()
            for i, j in ar_edges:
                (x0, y0), (x1, y1) = verts[i], verts[j]
                if -10000 < x0 < 10000 and -10000 < x1 < 10000 and -10000 < y0 < 10000 and -10000 < y1 < 10000:
                    cv.line(img2, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
            cv.imshow("AR", img2)
    


    cv.waitKey(50)
