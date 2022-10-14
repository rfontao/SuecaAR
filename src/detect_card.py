from card_identification import CardIdentifier
import cv2 as cv
import numpy as np
import sys
from camera import Camera
from card_detection import CardDetector
import imutils  # used to resize image

if (len(sys.argv) < 2):
    print(f"Usage: {sys.argv[0]} <file_to_read(.npz format)>")
    exit()

# Capture video from webcam
camera = Camera(sys.argv[2], sys.argv[1])
detector = CardDetector(True)
identifier = CardIdentifier()

while True:
    frame = camera.get_frame()
    cards = detector.detect(frame.copy())

    if len(cards) > 0:
        dst = np.array([
            [0, 0],
            [500, 0],
            [500, 726],
            [0, 726]
        ], dtype="float32")

        sortedPoints, center = detector.sortCardPoints(cards[0])
        # print(sortedPoints)
        # print("center", center)
        x, y = cards[0][0][0]
        x1, y1 = sortedPoints[0][0]
        x2, y2 = sortedPoints[1][0]
        x3, y3 = sortedPoints[2][0]
        x4, y4 = sortedPoints[3][0]

        # cv.circle(frame, [int(x), int(y)], 2, (0, 0, 0), 5)
        # cv.circle(frame, [int(x1), int(y1)], 2, (255, 0, 0), 2)
        # cv.circle(frame, [int(x2), int(y2)], 2, (0, 255, 0), 2)
        # cv.circle(frame, [int(x3), int(y3)], 2, (0, 0, 255), 2)
        # cv.circle(frame, [int(x4), int(y4)], 2, (255, 255, 255), 2)
        # cv.circle(frame, [int(center[0]), int(center[1])], 2, (255, 0, 255), 2)

        M = cv.getPerspectiveTransform(sortedPoints, dst)
        warp = cv.warpPerspective(frame, M, (500, 726))

        cv.imshow("Warp", warp)

        # cv.imshow("Warped", warp)
        template = warp[0:int(warp.shape[0]/4), 0:int(warp.shape[1]/6)]
        template = cv.GaussianBlur(template, (5, 5), 0)
        templateVal = warp[0:90, 0:90]
        templateType = warp[80:200, 0:90]

        cv.imshow("suit", templateType)
        cv.imshow("rank", templateVal)

        print(identifier.identify_rank(templateVal))
        print(identifier.identify_suit(templateType))

        cv.imshow("Original", frame)
        cv.imshow("Template", template)

    cv.waitKey(500)
