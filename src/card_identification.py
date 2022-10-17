import cv2 as cv


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
        contours, _ = cv.findContours(
            image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        if len(contours) > 1:
            x, y, w, h = cv.boundingRect(contours[1])
            image = image[y:y+h, x:x+w]

        image = cv.resize(
            image, (template.shape[1], template.shape[0]), interpolation=cv.INTER_LINEAR)
        
        cv.imshow("Template", image)

        result = cv.matchTemplate(image, template, cv.TM_SQDIFF_NORMED)
        _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(result)
        return _minVal

    def identify_rank(self, rank_image):
        scores = []
        rank_image = CardIdentifier.process_image_to_match(rank_image)
        cv.imshow("RANK_IMAGE", rank_image)
        for rank, template in self.rank_database:
            # cv.imshow("RANK", template)
            scores.append((rank, CardIdentifier.match_template(
                rank_image, template.copy())))

        scores = sorted(scores, key=lambda x: x[1])
        return scores

    def identify_suit(self, suit_image):
        scores = []
        suit_image = CardIdentifier.process_image_to_match(suit_image)
        cv.imshow("SUIT_IMAGE", suit_image)
        for suit, template in self.suit_database:
            scores.append((suit, CardIdentifier.match_template(
                suit_image, template.copy())))

        scores = sorted(scores, key=lambda x: x[1])
        return scores
