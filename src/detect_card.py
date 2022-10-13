from copyreg import constructor
from email.mime import image
import cv2 as cv
import numpy as np
import sys
from camera import Camera
from card_detection import CardDetector
import imutils  # used to resize image

if (len(sys.argv) < 2):
    print(f"Usage: {sys.argv[0]} <file_to_read(.npz format)>")
    exit()

# Capture video from webcam
camera = Camera(sys.argv[2], sys.argv[1])
detector = CardDetector(True)

cardimg = cv.imread("../cards/full/5.png", cv.IMREAD_COLOR)
cardimg = cv.cvtColor(cardimg, cv.COLOR_BGR2GRAY)
cardVal = cardimg[5:90, 10:90]  # gets the value of the card
cardType = cardimg[90:200, 10:90]  # gets the type of the card

cv.imshow("CardVal", cardVal)
cv.imshow("CardType", cardType)

while True:
    frame = camera.get_frame()
    # cards = detector.detect(frame.copy())

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cardType, None)
    kp2, des2 = sift.detectAndCompute(frame, None)

    # Initiate ORB detector
    # orb = cv.ORB_create()
    # # find the keypoints and descriptors with ORB
    # kp1, des1 = orb.detectAndCompute(cardVal, None)
    # kp2, des2 = orb.detectAndCompute(frame, None)

    FLANN_INDEX_KDTREE = 0
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # index_params = dict(algorithm=FLANN_INDEX_LSH,
    #                     table_number=6,  # 12
    #                     key_size=12,     # 20
    #                     multi_probe_level=1)  # 2
    search_params = dict(checks=50)
    # flann = cv.FlannBasedMatcher(index_params, search_params)
    # bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except:
        continue

    matches = list(filter(lambda x: len(x) == 2, matches))
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(cardType, kp1, frame, kp2,
                             matches, None, **draw_params)

    # matches = filter(lambda x: len(x) == 2, matches)
    # # store all the good matches as per Lowe's ratio test.
    # good = []
    # for m, n in matches:
    #     if m.distance < 0.7 * n.distance:
    #         good.append(m)

    # if len(good) > 10:
    #     src_pts = np.float32(
    #         [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    #     dst_pts = np.float32(
    #         [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    #     M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    #     matchesMask = mask.ravel().tolist()
    #     h, w = cardimg.shape
    #     pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
    #                      [w-1, 0]]).reshape(-1, 1, 2)
    #     dst = cv.perspectiveTransform(pts, M)
    #     frame = cv.polylines(frame, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    # else:
    #     print("Not enough matches are found - {}/{}".format(len(good), 10))
    #     matchesMask = None

    # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
    #                    singlePointColor=None,
    #                    matchesMask=matchesMask,  # draw only inliers
    #                    flags=2)
    # img3 = cv.drawMatches(cardimg, kp1, frame, kp2, good, None, **draw_params)

    cv.imshow("A", img3)

    # if len(cards) > 0:
    #     dst = np.array([
    #         [0, 0],
    #         [500, 0],
    #         [500, 726],
    #         [0, 726]], dtype="float32")

    #     M = cv.getPerspectiveTransform(detector.sortCardPoints(cards[0]), dst)
    #     warp = cv.warpPerspective(frame, M, (500, 726))
    #     cv.imshow("Original", frame)

    cv.waitKey(50)
