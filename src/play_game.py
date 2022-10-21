from pygame import Surface
from camera import Camera
import cv2 as cv
import numpy as np
from game_rules import Sueca, GameState
from model import Model
import sys

from card_identification import CardIdentifier
from card_detection import CardDetector

from opencv_renderer import OpenCVRenderer

if len(sys.argv) < 5:
    print(
        f"Usage: {sys.argv[0]} <file_to_read(.npz format)> <camera_feed> <number of game rounds> <trump_suit(Spades/Hearts/Diamonds/Clubs)>")
    exit()


def card_identification_process(frame, cards):
    final_suits = []
    final_ranks = []
    transforms = []

    dst = np.array([
        [0, 0],
        [500, 0],
        [500, 726],
        [0, 726]
    ], dtype="float32")

    aux = frame.copy()

    for card in cards:
        sortedPoints, center = detector.sortCardPoints(card)
        x1, y1 = sortedPoints[0][0]

        M = cv.getPerspectiveTransform(sortedPoints, dst)
        warp = cv.warpPerspective(frame, M, (500, 726))

        cv.imshow("Warped", warp)

        ret, ranks, suits = identifier.identify(warp)
        if ret:
            aux = identifier.show_predictions(
                aux, (int(x1), int(y1)), ranks, suits)
            if suits[0][1] < 0.5 and ranks[0][1] < 0.5:
                final_suits.append(suits[0][0])
                final_ranks.append(ranks[0][0])
                transforms.append(M)

    return aux, final_suits, final_ranks, transforms


# Capture video from webcam
camera = Camera(sys.argv[2], sys.argv[1])
detector = CardDetector(True)
identifier = CardIdentifier()
sueca = Sueca(sys.argv[3], sys.argv[4])

frames_between_rounds = 100
frame_counter = 0

USE_OPENGL = False

if USE_OPENGL:
    from OpenGL.GLUT import *
    from opengl_renderer import OpenGLRenderer

    renderer = OpenGLRenderer(
        camera.get_matrix(), int(camera.get_width()), int(camera.get_height()))
    # Models must be loaded after renderer is instantiated
    renderer.models.append(Model("../models/Cup1.obj", 0, swapyz=True,
                                 scale=[0.1, 0.1, 0.1]))
    renderer.models.append(Model("../models/Plastic_Cup.obj", 1,
                                 swapyz=True, scale=[0.03, 0.03, 0.03]))
else:
    renderer = OpenCVRenderer(
        camera.get_matrix(), camera.get_distortion_coeffs(), scale=0.1)

while True:
    frame = camera.get_frame()
    cards = detector.detect(frame.copy())

    if len(cards) > 0:
        frame, final_suits, final_ranks, transforms = card_identification_process(
            frame, cards)
        found_cards = [f"{r} {s}" for r, s in zip(final_suits, final_ranks)]

        if sueca.game_state == GameState.ROUND_RUNNING:
            sueca.register_cards(found_cards)

        # Draw winner on top of winning cards
        if sueca.game_state == GameState.ROUND_ENDED:
            frame_counter += 1
            if frame_counter == frames_between_rounds:
                print("ROUND STARTED")
                sueca.game_state = GameState.ROUND_RUNNING
            # frame = sueca.draw_winner_cards(frame, found_cards, transforms)

        sueca.draw_found_cards(frame)
        sueca.draw_team_scores(frame)

    if sueca.game_state == GameState.GAME_ENDED:
        renderer.draw_model = True
        renderer.picture = sueca.final_images[sueca.getWinner()]
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
                corners, 0.1, camera.get_matrix(), camera.get_distortion_coeffs())

        renderer.aruco_ids = ids
        renderer.tvecs = tvecs
        renderer.rvecs = rvecs

    renderer.image = frame.copy()
    renderer.display()

    cv.waitKey(50)
