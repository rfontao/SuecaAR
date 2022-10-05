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

while True:
    frame = camera.get_frame()
    cards = detector.detect(frame.copy())

    if len(cards) > 0:
        dst = np.array([
            [0, 0],
            [200, 0],
            [200, 400],
            [0, 400]], dtype="float32")

        # sorted(approx_list, key=lambda x:x[0][0]+x[0][1], reverse=True)
        M = cv.getPerspectiveTransform(cards[0], dst)
        warp = cv.warpPerspective(frame, M, (200, 400))
        cv.imshow("Warped", warp)

    cv.waitKey(50)
