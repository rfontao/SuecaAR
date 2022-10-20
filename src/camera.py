import cv2 as cv
import numpy as np


class Camera():

    def __init__(self, source, parameters_file):
        self.load_parameters(parameters_file)

        self.capture = cv.VideoCapture(int(source))
        if not self.capture.isOpened():
            print("Can't open video stream. Exiting...")
            exit()

    def load_parameters(self, parameters_file):
        parameters = np.load(parameters_file)
        self.matrix = parameters["camera_matrix"]
        self.dist_coeffs = parameters["distortion_coefficients"]

    def get_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            print("Can't receive frame (stream ended?). Exiting ...")
            exit()

        frame = cv.undistort(frame, self.matrix, self.dist_coeffs)

        return frame

    def get_width(self):
        return self.capture.get(cv.CAP_PROP_FRAME_WIDTH)

    def get_height(self):
        return self.capture.get(cv.CAP_PROP_FRAME_HEIGHT)

    def get_matrix(self):
        return self.matrix

    def get_distortion_coeffs(self):
        return self.dist_coeffs
