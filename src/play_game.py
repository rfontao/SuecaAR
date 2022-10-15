from card_identification import CardIdentifier
from game_rules import Sueca
import cv2 as cv
import numpy as np
import sys
from camera import Camera
from card_detection import CardDetector
import imutils  # used to resize image



# 0 - Find trump suit; 1 - playing game; 2 - declare winner
game_state = 0



def do_stuff(frame, card, dst):

    sortedPoints, center = detector.sortCardPoints(card)

    x, y = cards[0][0][0]
    x1, y1 = sortedPoints[0][0]
    x2, y2 = sortedPoints[1][0]
    x3, y3 = sortedPoints[2][0]
    x4, y4 = sortedPoints[3][0]

    M = cv.getPerspectiveTransform(sortedPoints, dst)
    warp = cv.warpPerspective(frame, M, (500, 726))

    cv.imshow("Warp", warp)

    # cv.imshow("Warped", warp)
    template = warp[0:int(warp.shape[0]/4), 0:int(warp.shape[1]/6)]
    template = cv.GaussianBlur(template, (5, 5), 0)
    templateVal = warp[0:90, 0:90]
    templateType = warp[80:200, 0:90]

    cv.imshow("suit", templateType)
    cv.imshow("rank", templateVal)

    values = identifier.identify_rank(templateVal)
    type = identifier.identify_suit(templateType)

    min_value = ("0", 15)
    min_type = ("A", 15)



    for i in range(0, len(values)):
        if(min_value[1] > values[i][1]):
            min_value = values[i]
    for i in range(0, len(type)):
        if(min_type[1] > type[i][1]):
            min_type = type[i]

    cv.imshow("Template", template)

    return min_value, min_type


if (len(sys.argv) < 3):
    print(f"Usage: {sys.argv[0]} <file_to_read(.npz format)> number_of_plays")
    exit()

sueca = Sueca(int(sys.argv[2]))

# Capture video from webcam
camera = Camera(sys.argv[2], sys.argv[1])
detector = CardDetector(True)
identifier = CardIdentifier()


while True:
    print(game_state)
    if game_state == 0:
        frame = camera.get_frame()
        cards = detector.detect(frame.copy())

        if len(cards) > 0:
            dst = np.array([
                [0, 0],
                [500, 0],
                [500, 726],
                [0, 726]
            ], dtype="float32")

            min_value, min_type = do_stuff(frame, cards[0], dst)

            if(min_value[1] < 0.35 and min_type[1] < 0.25):
                print("Trump found!")
                print("Value: " + str(min_value))
                print("Trump Type: " + str(min_type))
                sueca.trump_suit = min_type[0]
                game_state += 1
            else:
                print("Value: " + str(min_value))
                print("Trump Type: " + str(min_type))

            #print(identifier.identify_rank(templateVal))
            #print(identifier.identify_suit(templateType))

            cv.imshow("Original", frame)
            #cv.imshow("Template", template)

            

            #game_state += 1

    elif game_state == 1:
        frame = camera.get_frame()
        cards = detector.detect(frame.copy())

        if len(cards) > 3:
            dst = np.array([
                [0, 0],
                [500, 0],
                [500, 726],
                [0, 726]
            ], dtype="float32")


            min_value_c1, min_type_c1 = do_stuff(frame, cards[0], dst)
            min_value_c2, min_type_c2 = do_stuff(frame, cards[1], dst)
            min_value_c3, min_type_c3 = do_stuff(frame, cards[2], dst)
            min_value_c4, min_type_c4 = do_stuff(frame, cards[3], dst)

            card_1_valid = False
            card_2_valid = False
            card_3_valid = False
            card_4_valid = False

            print("Values: ")
            print([min_value_c1, min_type_c1])
            print([min_value_c2, min_type_c2])
            print([min_value_c3, min_type_c3])
            print([min_value_c4, min_type_c4])
            if(min_value_c1[1] < 0.6 and min_type_c1[1] < 0.5):
                print("Card 1 valid")
                card_1_valid = True
            if(min_value_c2[1] < 0.6 and min_type_c2[1] < 0.5):
                print("Card 2 valid")
                card_2_valid = True
            if(min_value_c3[1] < 0.6 and min_type_c3[1] < 0.5):
                print("Card 3 valid")
                card_3_valid = True
            if(min_value_c4[1] < 0.6 and min_type_c4[1] < 0.5):
                print("Card 4 valid")
                card_4_valid = True
            

            if (card_1_valid and card_2_valid and card_3_valid and card_4_valid):
                print("Cards found!")
                print("Values (in order): " + str([min_value_c1, min_value_c2, min_value_c3, min_value_c4]))
                print("Suits (in order): " + str([min_type_c1, min_type_c2, min_type_c3, min_type_c4]))
                #game_cards.append([min_value_c1, min_type_c1])
                #game_cards.append([min_value_c2, min_type_c2])
                #game_cards.append([min_value_c3, min_type_c3])
                #game_cards.append([min_value_c4, min_type_c4])
                sueca.calcRound(min_type_c1[0], min_value_c1[0], min_type_c2[0], min_value_c2[0], min_type_c3[0], min_value_c3[0], min_type_c4[0], min_value_c4[0])
                if(sueca.is_game_over()):
                    game_state += 1
                    
    elif game_state == 2:
        print("Game state = 2")
        print("Winner: " + str(sueca.getWinner()))
        #detect auruco and print winner

    cv.waitKey(2000)






