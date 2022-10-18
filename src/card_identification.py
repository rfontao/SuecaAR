import cv2 as cv
import numpy as np


class CardIdentifier():

    suit_paths = {
        "Spades": "1.png",
        "Hearts": "2.png",
        "Clubs": "3.png",
        "Diamonds": "4.png",
    }

    rank_paths = {
        "A": "1.png",    # A
        "2": "5.png",    # 2
        "3": "9.png",    # 3
        "4": "13.png",   # 4
        "5": "17.png",   # 5
        "6": "21.png",   # 6
        "7": "25.png",   # 7
        "Q": "41.png",   # Q
        "J": "45.png",   # J
        "K": "49.png",   # K
    }

    def __init__(self):
        self.load_suit_database()
        self.load_rank_database()

    def load_suit_database(self):
        self.suit_database = []
        for k, v in self.suit_paths.items():
            img = cv.imread(f"../cards/full/{v}", cv.IMREAD_GRAYSCALE)
            img = img[90:180, 10:90]  # Extracting the rank of the card
            _, img = cv.threshold(img, 180, 255, cv.THRESH_BINARY)
            self.suit_database.append((k, img))

    def load_rank_database(self):
        self.rank_database = []
        for k, v in self.rank_paths.items():
            img = cv.imread(f"../cards/full/{v}", cv.IMREAD_GRAYSCALE)
            img = img[15:85, 25:75]  # Extracting the suit of the card
            _, img = cv.threshold(img, 180, 255, cv.THRESH_BINARY)
            self.rank_database.append((k, img))

    @classmethod
    def process_image_to_match(cls, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.GaussianBlur(image, (5, 5), 0)
        _, image = cv.threshold(image, 180, 255, cv.THRESH_BINARY)
        # image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                              cv.THRESH_BINARY, 5, 2)

        # Test later with erosions (from: https://repositorio-aberto.up.pt/bitstream/10216/59981/1/000146528.pdf)
        # kernel = np.ones((3, 3), np.uint8)
        # templateVal = cv.erode(templateVal, kernel, iterations=1)
        return image

    @classmethod
    def match_template(cls, image, template):
        """Image should be previously processed with process_image_to_match"""
        # contours, _ = cv.findContours(
        #     image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        # contours = sorted(contours, key=cv.contourArea, reverse=True)
        # if len(contours) > 1:
        #     x, y, w, h = cv.boundingRect(contours[1])
        #     image = image[y:y+h, x:x+w]

        image = cv.resize(
            image, (template.shape[1], template.shape[0]), interpolation=cv.INTER_LINEAR)

        cv.imshow("Template", image)

        result = cv.matchTemplate(image, template, cv.TM_SQDIFF_NORMED)
        _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(result)
        return _minVal

    def extract_info(self, image):
        image = CardIdentifier.process_image_to_match(image)
        contours, _ = cv.findContours(
            image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        if len(contours) < 3:
            return None, None

        cpy = image.copy()

        x1, y1, w1, h1 = cv.boundingRect(contours[1])
        x2, y2, w2, h2 = cv.boundingRect(contours[2])

        if(w2 < 35):
            if len(contours) < 4:
                return None, None
            x2, y2, w2, h2 = cv.boundingRect(contours[3])

        # rank is always on top of suite
        if(y1 < y2):
            return image[y1:y1+h1, x1:x1+w1], image[y2:y2+h2, x2:x2+w2]
        else:
            return image[y2:y2+h2, x2:x2+w2], image[y1:y1+h1, x1:x1+w1]

    def identify_rank(self, rank_image):
        scores = []

        # rank_image = CardIdentifier.process_image_to_match(rank_image)
        cv.imshow("RANK_IMAGE", rank_image)
        for rank, template in self.rank_database:
            scores.append([rank, CardIdentifier.match_template(
                rank_image, template.copy())])

        scores = sorted(scores, key=lambda x: x[1])
        return scores

    def identify_suit(self, suit_image, is_red):
        scores = []

        # Must be before processing
        # is_red = CardIdentifier.is_suit_red(suit_image)

        # suit_image = CardIdentifier.process_image_to_match(suit_image)
        cv.imshow("SUIT_IMAGE", suit_image)
        for suit, template in self.suit_database:
            scores.append([suit, CardIdentifier.match_template(
                suit_image, template.copy())])

        for score in scores:
            # Lower score is better
            if (score[0] == "Spades" or score[0] == "Clubs") and is_red:
                score[1] = 2.0
            if (score[0] == "Hearts" or score[0] == "Diamonds") and not is_red:
                score[1] = 2.0

        scores = sorted(scores, key=lambda x: x[1])
        return scores

    @classmethod
    def is_suit_red(cls, image):
        """Returns if the suit in image is red (Hearts or Diamonds)"""
        img = image.copy()
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # Taken from https://stackoverflow.com/questions/30331944/finding-red-color-in-image-using-python-opencv
        # lower mask (0-10)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask0 = cv.inRange(hsv, lower_red, upper_red)

        # upper mask (170-180)
        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask1 = cv.inRange(hsv, lower_red, upper_red)

        # join my masks
        mask = mask0 + mask1

        # set my output img to zero everywhere except my mask
        img = cv.bitwise_and(img, img, mask=mask)

        sparsity = 1.0 - (np.count_nonzero(img) / float(img.size))
        return sparsity < 0.9

        # cv.imshow("IS_RED?", out)

    def identify(self, image):
        template = image[0:int(image.shape[0] / 4), 0:int(image.shape[1] / 6)]
        template = cv.GaussianBlur(template, (5, 5), 0)
        template = image[0:200, 0:90]
        template = cv.copyMakeBorder(
            template, 10, 10, 10, 10, cv.BORDER_CONSTANT, None, value=(255, 255, 255))

        is_red = CardIdentifier.is_suit_red(template)

        rank_img, suit_img = self.extract_info(template)

        if rank_img is None and suit_img is None:
            return False, [], []

        cv.imshow("suit", suit_img)
        cv.imshow("rank", rank_img)

        ranks = self.identify_rank(rank_img)
        suits = self.identify_suit(suit_img, is_red)

        return True, ranks, suits

    def show_predictions(self, image, point, ranks, suits):
        print(ranks)
        print(suits)
        suit_text = f"Suit: {suits[0][0]} - {round(suits[0][1], 2)}" if suits[0][1] < 0.5 else "Can't determine suit"
        rank_text = f"Rank: {ranks[0][0]} - {round(ranks[0][1], 2)}" if ranks[0][1] < 0.5 else "Can't determine rank"

        cv.putText(
            image,
            suit_text,
            point,
            cv.FONT_HERSHEY_COMPLEX,
            1.0,
            (255, 255, 255)
        )
        cv.putText(
            image,
            rank_text,
            (point[0], point[1] - 30),
            cv.FONT_HERSHEY_COMPLEX,
            1.0,
            (255, 255, 255)
        )

        return image
