import cv2 as cv
import numpy as np
import sys
from camera import Camera
from card_detection import CardDetector

if (len(sys.argv) < 2):
    print(f"Usage: {sys.argv[0]} <file_to_read(.npz format)>")
    exit()

# Capture video from webcam
camera = Camera(0, sys.argv[1])
detector = CardDetector(True)

template = cv.imread("../cards/simple/4d.jpg", cv.IMREAD_COLOR)

while True:
    frame = camera.get_frame()
    cards = detector.detect(frame.copy())

    if len(cards) > 0:
        dst = np.array([
            [0, 0],
            [200, 0],
            [200, 400],
            [0, 400]], dtype="float32")

        M = cv.getPerspectiveTransform(cards[0], dst)
        warp = cv.warpPerspective(frame, M, (200, 400))
        cv.imshow("Warped", warp)

        method = cv.TM_CCOEFF_NORMED
        result = cv.matchTemplate(warp, template, method)
        cv.normalize(result, result, 0, 1, cv.NORM_MINMAX, -1)
        _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(result, None)
        if (method == cv.TM_SQDIFF or method == cv.TM_SQDIFF_NORMED):
            matchLoc = minLoc
        else:
            matchLoc = maxLoc

        cv.rectangle(frame, matchLoc, (matchLoc[0] + template.shape[0],
                     matchLoc[1] + template.shape[1]), (0, 0, 0), 2, 8, 0)
        cv.rectangle(result, matchLoc, (matchLoc[0] + template.shape[0],
                     matchLoc[1] + template.shape[1]), (0, 0, 0), 2, 8, 0)

                    
        cv.imshow("Original", frame)
        cv.imshow("Matching", result)

    cv.waitKey(50)
