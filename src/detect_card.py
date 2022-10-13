from copyreg import constructor
from email.mime import image
import cv2 as cv
import numpy as np
import sys
from camera import Camera
from card_detection import CardDetector
import imutils # used to resize image

if (len(sys.argv) < 2):
    print(f"Usage: {sys.argv[0]} <file_to_read(.npz format)>")
    exit()

# Capture video from webcam
camera = Camera(sys.argv[2], sys.argv[1])
detector = CardDetector(True)

cardimg = cv.imread("../cards/full/5.png", cv.IMREAD_COLOR)
cardVal = cardimg[5:90, 10:90] # gets the value of the card
cardType = cardimg[90:200, 10:90] # gets the type of the card

cv.imshow("CardVal", cardVal)
cv.imshow("CardType", cardType)

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
        print(sortedPoints)
        print("center", center)
        x, y = cards[0][0][0]
        x1, y1 = sortedPoints[0][0]
        x2, y2 = sortedPoints[1][0]
        x3, y3 = sortedPoints[2][0]
        x4, y4 = sortedPoints[3][0]

        cv.circle(frame, [int(x), int(y)], 2, (0,0,0), 5)
        cv.circle(frame, [int(x1), int(y1)], 2, (255,0,0), 2)
        cv.circle(frame, [int(x2), int(y2)], 2, (0,255,0), 2)
        cv.circle(frame, [int(x3), int(y3)], 2, (0,0,255), 2)
        cv.circle(frame, [int(x4), int(y4)], 2, (255,255,255), 2)
        cv.circle(frame, [int(center[0]), int(center[1])], 2, (255,0,255), 2)


        M = cv.getPerspectiveTransform(sortedPoints, dst)
        warp = cv.warpPerspective(frame, M, (500, 726))

        cv.imshow("Warp", warp)


        # cv.imshow("Warped", warp)
        template = warp[0:int(warp.shape[0]/4), 0:int(warp.shape[1]/6)]

        template = cv.GaussianBlur(template, (5, 5), 0)

        # templateVal = warp[0:50, 0:40]
        # templateType = warp[50:100, 0:40]

        cv.imshow("template", template)
        # cv.imshow("Val", templateVal)
        # cv.imshow("Type", templateType)

        method = cv.TM_CCOEFF_NORMED
    
        resultType = cv.matchTemplate(template, cardimg, method)
        # resultVal = cv.matchTemplate(templateVal, cardimg, method)

        # cv.normalize(result, result, 0, 1, cv.NORM_MINMAX, -1)
        # cv.imshow("Template", template)            
        cv.imshow("Original", frame)
        # cv.imshow("Matching", resultType)
        # cv.imshow("Match", copy)

        _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(resultType)
        if(_maxVal < 0.9):
            print("Not Match Type: " + str(_maxVal))
            cv.waitKey(1000)
            continue

        # _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(resultVal)
        # if(_maxVal < 0.8):
        #     print("Not Match Val " + str(_maxVal))
        #     cv.waitKey(1000)
        #     continue
        
        print("Match " + str(_maxVal))

        if (method == cv.TM_SQDIFF or method == cv.TM_SQDIFF_NORMED):
            matchLoc = minLoc
        else:
            matchLoc = maxLoc
        
        copy = cardimg.copy()

        cv.rectangle(copy, matchLoc, (matchLoc[0] + template.shape[0],
                    matchLoc[1] + copy.shape[1]), 255, 2)

        
    cv.waitKey(1000)
