from dataclasses import dataclass
import cv2 as cv


class Sueca:
    CARD_TYPE_TO_POINTS = {
        "A": 11,
        "7": 10,
        "K": 4,
        "J": 3,
        "Q": 2,
        "6": 0,
        "5": 0,
        "4": 0,
        "3": 0,
        "2": 0,
        "10": -121,
        "9": -121,
        "8": -121
    }

    CARD_TYPE_PRIORITY = {
        "A": 9,
        "7": 8,
        "K": 7,
        "J": 6,
        "Q": 5,
        "6": 4,
        "5": 3,
        "4": 2,
        "3": 1,
        "2": 0,
        "10": -121,
        "9": -121,
        "8": -121
    }

    team_1_points = 0
    team_2_points = 0

    cards_played = 0

    trump_suit = ""

    total_number_rounds = 0

    play_cards_suits = []
    play_cards_values = []

    current_round_cards = {}
    fixed_current_round_cards = []
    current_round = 0

    # TRUMP_SUITS = ["Spades", "Hearts", "Clubs", "Diamonds"] #Copas, Espadas, Ouros, Paus

    def __init__(self, number_rounds, trump_suit):
        self.vertical_team_points = 0
        self.horizontal_team_points = 0
        self.cards_played = 0
        self.total_number_rounds = int(number_rounds)
        self.game_state = 0
        self.setTrumpSuit(trump_suit)
        self.final_images = [cv.imread("../cards/player_1_win.png"), cv.imread(
            "../cards/player_2_win.png"), cv.imread("../cards/tie.png")]

    def is_game_over(self):
        return self.total_number_rounds <= self.cards_played / 4

    def setTrumpSuit(self, trump_suit):
        self.trump_suit = trump_suit
        print(f"Set trump suit: {trump_suit}")

    def getWinner(self):
        if not self.is_game_over:
            return -1
        if self.team_1_points > self.team_2_points:
            return 1
        elif self.team_2_points > self.team_1_points:
            return 2
        return 0
        # if(self.team_1_points > 60):
        #    return 1
        # if(self.team_2_points > 60):
        #    return 2

    def register_cards(self, cards):

        # self.current_round_cards = list(
        #     filter(lambda x: x[0] in cards, self.current_round_cards))
        cards = list(
            filter(lambda x: x not in self.fixed_current_round_cards, cards))
        self.current_round_cards = {key: value for (key, value) in self.current_round_cards.items() if key in cards}

        for k in self.current_round_cards.keys():
            if k in cards:
                self.current_round_cards[k] += 1

        # Add new cards found
        for c in cards:
            if c not in self.current_round_cards:
                self.current_round_cards[c] = 0

        # If a card has been seen n frames mark it as present
        for k, v in self.current_round_cards.items():
            if v >= 5:
                self.fixed_current_round_cards.append(k)

        if len(self.fixed_current_round_cards) == 4:
            self.calcRound()

    def calcRound(self):

        cards = self.fixed_current_round_cards
        print(cards)

        # Calculate winner
        priorities = [(999 if c.split(" ")[0] == self.trump_suit else 0) +
                      self.CARD_TYPE_PRIORITY[c.split(" ")[1]] for c in cards]
        max_prio = max(priorities)

        points = [self.CARD_TYPE_TO_POINTS[c.split(" ")[1]] for c in cards]
        roundPoints = sum(points)

        # Add points to winner's team
        if priorities[0] == max_prio or priorities[2] == max_prio:
            self.team_1_points += roundPoints
        else:
            self.team_2_points += roundPoints

        if self.current_round == self.total_number_rounds:
            self.is_game_over = True
            self.game_state = 1

        self.current_round += 1
        self.current_round_cards = {}
        self.fixed_current_round_cards = []

    def draw_found_cards(self, frame):

        # print(f"Found:{self.current_round_cards}")
        # print(f"Fixed Found:{self.fixed_current_round_cards}")
        cur_y = 20
        for c in self.fixed_current_round_cards:
            text = f"Found card: {c}"

            cv.putText(
                frame,
                text,
                [5, cur_y],
                cv.FONT_HERSHEY_COMPLEX,
                1.0,
                (0, 255, 255)
            )

            cur_y += 30

    # def tryAddCards(self, list_suits, list_values):
    #     for i in range(0, len(list_suits)):
    #         already_in_list = False
    #         for j in range(len(self.play_cards_suits)):
    #             if list_suits[i] == self.play_cards_suits[j] and list_values[i] == self.play_cards_values[j]:
    #                 already_in_list = True
    #         if not already_in_list:
    #             self.play_cards_suits.append(list_suits[i])
    #             self.play_cards_values.append(list_values[i])

    #     if len(self.play_cards_suits) >= 4:
    #         self.calcRound(self.play_cards_suits[0], self.play_cards_values[0], self.play_cards_suits[1], self.play_cards_values[1],
    #                        self.play_cards_suits[2], self.play_cards_values[2], self.play_cards_suits[3], self.play_cards_values[3])
    #         self.play_cards_suits.clear()
    #         self.play_cards_values.clear()
