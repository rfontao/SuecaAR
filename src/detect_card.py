from copyreg import constructor
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

template = cv.imread("../cards/full/1.png", cv.IMREAD_COLOR)
template = template[0:200, 0:100]
templateVal = template[5:90, 10:900] # gets the value of the card
templateType = template[90:200, 10:90] # gets the type of the card

cv.imshow("TemplateVal", templateVal)
cv.imshow("TemplateType", templateType)

while True:
    frame = camera.get_frame()
    cards = detector.detect(frame.copy())

    if len(cards) > 0:
        dst = np.array([
            [0, 0],
            [200, 0],
            [200, 400],
            [0, 400]], dtype="float32")

        M = cv.getPerspectiveTransform(detector.sortCardPoints(cards[0]), dst)
        warp = cv.warpPerspective(frame, M, (200, 400))

        # cv.imshow("Warped", warp)
        cropped = warp[0:int(warp.shape[0]/4), 0:int(warp.shape[1]/6)]

        cropped = cv.GaussianBlur(cropped, (5, 5), 0)

        croppedVal = warp[0:50, 0:40]
        croppedType = warp[50:100, 0:40]

        cv.imshow("Cropped", cropped)
        cv.imshow("Val", croppedVal)
        cv.imshow("Type", croppedType)

        method = cv.TM_CCOEFF_NORMED
    
        resultType = cv.matchTemplate(croppedType, templateType, method)
        resultVal = cv.matchTemplate(croppedVal, templateVal, method)

        # cv.normalize(result, result, 0, 1, cv.NORM_MINMAX, -1)
        _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(resultType)
        if(_maxVal < 0.6):
            print("Not Match Type: " + str(_maxVal))
            cv.waitKey(1000)
            continue

        _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(resultVal)
        if(_maxVal < 0.6):
            print("Not Match Val " + str(_maxVal))
            cv.waitKey(1000)
            continue
        
        print("Match " + str(_maxVal))

        # if (method == cv.TM_SQDIFF or method == cv.TM_SQDIFF_NORMED):
        #     matchLoc = minLoc
        # else:
        #     matchLoc = maxLoc
        
        # cv.rectangle(template, matchLoc, (matchLoc[0] + template.shape[0],
        #             matchLoc[1] + template.shape[1]), 255, 2)

        cv.imshow("Template", template)            
        cv.imshow("Original", frame)
        cv.imshow("Matching", resultType)

    cv.waitKey(1000)
