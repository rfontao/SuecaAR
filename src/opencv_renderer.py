import cv2 as cv
import numpy as np


class openCVRenderer():

    ar_verts = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                           [0, 0, 0.75], [0, 1, 0.75], [1, 1, 0.75], [1, 0, 0.75]])
    ar_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)]

    def __init__(self, camera_matrix, dist_coeffs, scale=1, window_name="SuecaAR"):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.window_name = window_name

        def translateXY(x):
            x[0] -= 0.5
            x[1] -= 0.5
            return x

        self.ar_verts = np.float32(list(map(translateXY, self.ar_verts)))
        self.ar_verts *= scale
        cv.namedWindow(self.window_name)

    def display(self, image, rvecs, tvecs):
        for rvec in rvecs:
            for tvec in tvecs:
                verts = cv.projectPoints(self.ar_verts, rvec, tvec, self.camera_matrix, self.dist_coeffs)[
                    0].reshape(-1, 2)
                for i, j in self.ar_edges:
                    (x0, y0), (x1, y1) = verts[i], verts[j]
                    if -10000 < x0 < 10000 and -10000 < x1 < 10000 and -10000 < y0 < 10000 and -10000 < y1 < 10000:
                        cv.line(image, (int(x0), int(y0)),
                                (int(x1), int(y1)), (0, 0, 255), 2)

        cv.imshow(self.window_name, image)
