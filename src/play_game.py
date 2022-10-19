from camera import Camera
import cv2 as cv
import numpy as np
from game_rules import Sueca
from model import Model
import sys

from card_identification import CardIdentifier
from card_detection import CardDetector
import imutils  # used to resize image

from opencv_renderer import OpenCVRenderer

if (len(sys.argv) < 3):
    print(f"Usage: {sys.argv[0]} <file_to_read(.npz format)> <number of game rounds>")
    exit()

sueca = Sueca(sys.argv[2])


def card_identification_process(frame, cards):
    final_suits = []
    final_ranks = []

    for card in cards:

        sortedPoints, center = detector.sortCardPoints(card)
        # print(sortedPoints)
        # print("center", center)
        x, y = card[0][0]
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

        cv.imshow("Warped", warp)

        ret, ranks, suits = identifier.identify(warp)
        if ret:
            frame = identifier.show_predictions(frame, (int(x1), int(y1)), ranks, suits)
            if suits[0][1] < 0.5 and ranks[0][1] < 0.5:
                final_suits.append(suits[0][0])
                final_ranks.append(ranks[0][0])


    return final_suits, final_ranks

sueca.game_state = 0

# Capture video from webcam
camera = Camera(sys.argv[2], sys.argv[1])
detector = CardDetector(True)
identifier = CardIdentifier()


wait_more_time = False

while sueca.game_state < 2:
    frame = camera.get_frame()
    cards = detector.detect(frame.copy())

    if len(cards) > 0:
        dst = np.array([
            [0, 0],
            [500, 0],
            [500, 726],
            [0, 726]
        ], dtype="float32")

        final_suits, final_ranks = card_identification_process(frame, cards)

        wait_more_time = False

        if sueca.game_state == 0:
            if(len(final_suits) > 0):
                sueca.setTrumpSuit(final_suits[0])
                wait_more_time = True
        elif sueca.game_state == 1:
            for i in range(0, len(final_suits)):
                # Add suits and ranks to game, as long as they don't exceed 4 at once
                sueca.tryAddCards(final_suits, final_ranks)
                if sueca.game_state != 2:
                    wait_more_time = True
        cv.imshow("Original", frame)
        # cv.imshow("Template", template)

    if (wait_more_time):
        cv.waitKey(2000)
    else:
        cv.waitKey(50)

while sueca.game_state >= 2:
    
    USE_OPENGL = False

    if USE_OPENGL:
        from OpenGL.GLUT import *
        from opengl_renderer import OpenGLRenderer

    files = np.load("../params_1.npz")
    camera_matrix = files["camera_matrix"]
    distortion_coefficient = files["distortion_coefficients"]


    camera = Camera(1, sys.argv[1])
    if USE_OPENGL:
        renderer = OpenGLRenderer(
            camera_matrix, int(camera.get_width()), int(camera.get_height()))
        # Models must be loaded after renderer is instantiated
        renderer.models.append(Model("../models/Cup1.obj", 0, swapyz=True,
                                    scale=[0.1, 0.1, 0.1]))
        renderer.models.append(Model("../models/Plastic_Cup.obj", 1,
                                    swapyz=True, scale=[0.03, 0.03, 0.03]))
    else:
        renderer = OpenCVRenderer(camera_matrix, distortion_coefficient, scale=0.1)
        if sueca.getWinner() == 1:
            renderer.picture = cv.imread("../cards/player_1_win.png")
        elif sueca.getWinner() == 2:
            renderer.picture = cv.imread("../cards/player_2_win.png")
        else:
            renderer.picture = cv.imread("../cards/tie.png")



    while True:

        frame = camera.get_frame()

        rvecs = []
        tvecs = []


        arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
        arucoParams = cv.aruco.DetectorParameters_create()
        corners, ids, rejected = cv.aruco.detectMarkers(frame, arucoDict,
                                                        parameters=arucoParams)
        if len(corners) == 0:
            ids = []

        if len(corners) > 0:
            cv.aruco.drawDetectedMarkers(frame, corners, ids)

            rvecs, tvecs, objPoints = cv.aruco.estimatePoseSingleMarkers(
                corners, 0.1, camera_matrix, distortion_coefficient)

            # for i in range(len(rvecs)):
            #     img = cv.drawFrameAxes(
            #         img, camera_matrix, distortion_coefficient, rvecs[i], tvecs[i], 0.05)


        renderer.aruco_ids = ids
        renderer.tvecs = tvecs
        renderer.rvecs = rvecs
        renderer.image = frame.copy()
        renderer.display()

        cv.waitKey(50)






