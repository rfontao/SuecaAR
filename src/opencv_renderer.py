import cv2 as cv
import numpy as np


class OpenCVRenderer():

    ar_verts = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                           [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]])
    ar_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)]

    # image_points = np.float32([[1, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1]])
    image_points = np.float32([[0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1]])

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
        self.image_points = np.float32(
            list(map(translateXY, self.image_points)))
        self.image_points *= scale

        self.picture = cv.imread("../cards/full/1.png")

        self.image = None
        self.aruco_ids = []
        self.rvecs = []
        self.tvecs = []
        self.display_models = False
        cv.namedWindow(self.window_name)

    def display_wireframes(self):
        for rvec in self.rvecs:
            for tvec in self.tvecs:
                verts = cv.projectPoints(self.ar_verts, rvec, tvec, self.camera_matrix, self.dist_coeffs)[
                    0].reshape(-1, 2)
                for i, j in self.ar_edges:
                    (x0, y0), (x1, y1) = verts[i], verts[j]
                    cv.line(self.image, (int(x0), int(y0)),
                            (int(x1), int(y1)), (0, 0, 255), 2)

    def display_image(self):
        for rvec in self.rvecs:
            for tvec in self.tvecs:
                verts = cv.projectPoints(self.image_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)[
                    0].reshape(-1, 2)

                src_points = np.float32([(0.0, 0.0), (self.picture.shape[1]-1.0, 0.0),
                                         (self.picture.shape[1]-1.0, self.picture.shape[0]-1.0), (0.0, self.picture.shape[0]-1.0)])
                dst_points = verts

                M = cv.getPerspectiveTransform(src_points, dst_points)
                warp = cv.warpPerspective(
                    self.picture, M, (self.image.shape[1], self.image.shape[0]))

                dst_points = np.int32(dst_points)
                self.image = cv.fillConvexPoly(self.image, dst_points, 0)
                self.image = self.image + warp

    def display(self):
        if self.display_models:
            self.display_wireframes()
            self.display_image()

        cv.imshow(self.window_name, self.image)
