
from audioop import cross
from copyreg import constructor
from dis import dis
from time import sleep
from xmlrpc.client import MAXINT
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

    def lineIntersection(self, pair1, pair2):
        pairs = np.vstack([pair1, pair2])
        pairs = np.hstack([pairs, np.ones((4,1))])
        l1 = np.cross(pairs[0], pairs[1])
        l2 = np.cross(pairs[2], pairs[3])
        x, y, z = np.cross(l1, l2)
        if(z == 0):
            return np.inf, np.inf
        return x/z, y/z

    def inSecondAndThirdQuadrant(self, angle1, angle2):
        return angle1 > (np.pi/2) and angle2 < (-np.pi/2)

    def sortCardPoints(self, cardPoints):
        distDict = {}
        distDict[np.linalg.norm(cardPoints[0][0] - cardPoints[1][0])] = cardPoints[1]
        distDict[np.linalg.norm(cardPoints[0][0] - cardPoints[2][0])] = cardPoints[2]
        distDict[np.linalg.norm(cardPoints[0][0] - cardPoints[3][0])] = cardPoints[3]

        sortedDists = sorted(distDict)
        pair1 = [cardPoints[0][0], distDict[sortedDists[2]][0]]
        pair2 = [distDict[sortedDists[0]][0], distDict[sortedDists[1]][0]]

        x, y = self.lineIntersection(pair1, pair2)
        
        center = [x,y]
        vec1 = cardPoints[0][0] - center
        vec2 = distDict[sortedDists[0]][0] - center
        angle1 = np.arctan2(-vec1[1], vec1[0])
        angle2 = np.arctan2(-vec2[1], vec2[0])


        if(self.inSecondAndThirdQuadrant(angle1, angle2)):
            angle2 = angle2 + 2*np.pi
        elif((angle1 > -np.pi and angle1 < -np.pi/2) or (angle2 > -np.pi and angle2 < -np.pi/2)):
            angle1 = (angle1 + np.pi)
            angle2 = (angle2 + np.pi)
        elif((angle1 > -np.pi/2 and angle1 < 0) or (angle2 > -np.pi/2 and angle2 < 0)):
            angle1 = angle1 + np.pi/2
            angle2 = angle2 + np.pi/2
        


        if(angle1 > angle2):
            return (np.array([cardPoints[0], distDict[sortedDists[0]], distDict[sortedDists[2]], distDict[sortedDists[1]]]), center)
        else:
            return (np.array([ distDict[sortedDists[0]], cardPoints[0], distDict[sortedDists[1]],  distDict[sortedDists[2]]]), center)