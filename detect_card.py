import cv2 as cv
import math

cv.namedWindow("Webcam")
cv.namedWindow("Edges")


t1 = 100
t2 = 200

# Capture video from webcam
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Can't open stream")
    exit()

while True:
    ret, img = cap.read()

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
        if hierarchy[0, i , 3] != -1:
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

    img1 = img.copy()
    cv.drawContours(img, contours, -1, (0, 255, 0), 3)
    cv.drawContours(img1, approx_list, -1, (255, 0, 0), 3)

    cv.imshow("Webcam", img)
    cv.imshow("Contours", img1)

    cv.waitKey(10)
