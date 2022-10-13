
from copyreg import constructor
from dis import dis
from time import sleep
import cv2 as cv
import numpy as np


class CardDetector():

    def __init__(self, debug=False):
        self.t1 = 200
        self.t2 = 255
        self.area_threshold = 1000

        self.debug = debug
        if self.debug:
            cv.namedWindow("Card detection")
            cv.createTrackbar("Threshold 1", "Card detection",
                              self.t1, 255, self.changeThreshold1Callback)
            cv.createTrackbar("Threshold 2", "Card detection",
                              self.t2, 255, self.changeThreshold2Callback)
            cv.createTrackbar("Area Threshold", "Card detection",
                              self.area_threshold, 1000, self.changeAreaThresholdCallback)

    def detect(self, image):

        # Converting to grayscale and applying Gaussian blur
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)

        # Applying CLAHE histogram equalization
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(blur)

        # Applying threshold for edge detection
        _, edges = cv.threshold(equalized, self.t1, self.t2, cv.THRESH_BINARY)

        # Finding contours with RETR_EXTERNAL (does not include elements inside the card)
        contours, _ = cv.findContours(
            edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        matches = []
        for c in contours:
            perimeter = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.04 * perimeter, True)

            # Check if it's convex
            if not cv.isContourConvex(approx):
                continue


            # Threshold area
            area = cv.contourArea(approx)
            if area < self.area_threshold:
                continue

            # Check if contour has 4 sides
            if len(approx) == 4:
                matches.append(approx)

        # Sort by biggest area
        matches = sorted(matches, key=cv.contourArea, reverse=True)

        # If debug is enabled show steps
        if self.debug:
            contour_image = image.copy()
            cv.drawContours(contour_image, matches, -1, (0, 255, 0), 3)

            edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
            debug_image = cv.hconcat([edges, contour_image])

            cv.imshow("Card detection", debug_image)

        return np.array(matches, dtype=np.float32)

    def changeThreshold1Callback(self, val):
        self.t1 = val

    def changeThreshold2Callback(self, val):
        self.t2 = val
    
    def changeAreaThresholdCallback(self, val):
        self.area_threshold = val

    def sortCardPoints(self, cardPoints):
        dist1 = np.linalg.norm(cardPoints[0] - cardPoints[1])
        dist2 = np.linalg.norm(cardPoints[0] - cardPoints[2])
        dist3 = np.linalg.norm(cardPoints[0] - cardPoints[3])
        sortedDists = sorted([dist1, dist2, dist3])
        ret = np.array([cardPoints[0]])
        
        if(dist1 == sortedDists[0]):
            if(cardPoints[1][0][0] < cardPoints[0][0][0]):
                cardPoints[0], cardPoints[1] = np.copy(cardPoints[1]), np.copy(cardPoints[0])
                return self.sortCardPoints(cardPoints)
            ret = np.append(ret, [cardPoints[1]], axis=0)
            if(dist2 == sortedDists[1]):
                ret = np.append(ret, [cardPoints[3], cardPoints[2]], axis=0)
            else:
                ret = np.append(ret, [cardPoints[2], cardPoints[3]], axis=0)

        elif(dist2 == sortedDists[0]):

            if(cardPoints[2][0][0] < cardPoints[0][0][0]):
                cardPoints[0], cardPoints[2] = np.copy(cardPoints[2]), np.copy(cardPoints[0])
                return self.sortCardPoints(cardPoints)

            ret = np.append(ret, [cardPoints[2]], axis=0)
            if(dist1 == sortedDists[1]):
                ret = np.append(ret, [cardPoints[3], cardPoints[1]], axis=0)
            else:
                ret = np.append(ret, [cardPoints[1], cardPoints[3]], axis=0)

        elif(dist3 == sortedDists[0]):

            if(cardPoints[3][0][0] < cardPoints[0][0][0]):
                cardPoints[0], cardPoints[3] = np.copy(cardPoints[3]), np.copy(cardPoints[0])
                return self.sortCardPoints(cardPoints)

            ret = np.append(ret, [cardPoints[3]], axis=0)
            if(dist1 == sortedDists[1]):
                ret = np.append(ret, [cardPoints[2], cardPoints[1]], axis=0)
            else:
                ret = np.append(ret, [cardPoints[1], cardPoints[2]], axis=0)
        
        return ret