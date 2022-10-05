import cv2 as cv
import numpy as np


class CardDetector():

    def __init__(self, debug=False):
        self.debug = debug
        if self.debug:
            cv.namedWindow("Card detection")

    def detect(self, image):

        # Converting to grayscale and applying Gaussian blur
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)

        # Applying CLAHE histogram equalization
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(blur)

        # Applying threshold for edge detection
        _, edges = cv.threshold(equalized, 200, 255, cv.THRESH_BINARY)

        # Finding contours with RETR_EXTERNAL (does not include elements inside the card)
        contours, _ = cv.findContours(
            edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        matches = []
        for c in contours:
            perimeter = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.04 * perimeter, True)

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
