from dataclasses import dataclass
import cv2 as cv
from enum import Enum
import numpy as np


class GameState(Enum):
    GAME_ENDED = 1
    ROUND_RUNNING = 2
    ROUND_ENDED = 3


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

    trump_suit = ""

    total_number_rounds = 0

    current_round_cards = {}
    fixed_current_round_cards = []
    winning_cards = []
    current_round = 0

    player_who_won = 1  # 1, 2, 3, 4; first round -> 1

    # TRUMP_SUITS = ["Spades", "Hearts", "Clubs", "Diamonds"] #Copas, Espadas, Ouros, Paus

    def __init__(self, number_rounds, trump_suit):
        self.vertical_team_points = 0
        self.horizontal_team_points = 0
        self.total_number_rounds = int(number_rounds)
        self.game_state = GameState.ROUND_RUNNING
        self.setTrumpSuit(trump_suit)
        self.final_images = [cv.imread("../cards/player_1_win.png"), cv.imread(
            "../cards/player_2_win.png"), cv.imread("../cards/tie.png")]

    def is_game_over(self):
        return self.current_round == self.total_number_rounds

    def start_new_round(self):
        self.winning_cards = []
        self.game_state = GameState.ROUND_RUNNING

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
        self.current_round_cards = {key: value for (
            key, value) in self.current_round_cards.items() if key in cards}

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

        player_one = self.player_who_won - 1
        player_two = (self.player_who_won + 1) % 4 - 1
        player_three = (self.player_who_won + 2) % 4 - 1
        player_four = (self.player_who_won + 3) % 4 - 1

        if self.player_who_won % 2 == 0:
            player_one = (player_one + 2) % 4
            player_two = (player_two + 2) % 4
            player_three = (player_three + 2) % 4
            player_four = (player_four + 2) % 4

        if player_one < 0:
            player_one += 4
        if player_two < 0:
            player_two += 4
        if player_three < 0:
            player_three += 4
        if player_four < 0:
            player_four += 4

        # Add points to winner's team
        if priorities[player_one] == max_prio or priorities[player_three] == max_prio:
            self.team_1_points += roundPoints
            self.winning_cards = [cards[player_one], cards[player_three]]
            if priorities[player_one] == max_prio:
                self.player_who_won = 1
            if priorities[player_three] == max_prio:
                self.player_who_won = 3
        else:
            self.team_2_points += roundPoints
            self.winning_cards = [cards[player_two], cards[player_four]]
            if priorities[player_two] == max_prio:
                self.player_who_won = 2
            if priorities[player_four] == max_prio:
                self.player_who_won = 4

        print("Player who won:")
        print(self.player_who_won)
        self.current_round += 1

        if self.current_round == self.total_number_rounds:
            self.is_game_over = True
            self.game_state = GameState.GAME_ENDED
            print("GAME ENDED")
        else:
            self.game_state = GameState.ROUND_ENDED
            print("ROUND_ENDED")

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
                0.6,
                (0, 255, 255)
            )

            cur_y += 20

    def draw_team_scores(self, frame):

        cur_y = frame.shape[0] - 30

        cv.putText(
            frame,
            f"Team 1 score: {self.team_1_points}",
            [5, cur_y],
            cv.FONT_HERSHEY_COMPLEX,
            0.7,
            (255, 255, 0)
        )

        cv.putText(
            frame,
            f"Team 2 score: {self.team_2_points}",
            [5, cur_y + 15],
            cv.FONT_HERSHEY_COMPLEX,
            0.7,
            (255, 255, 0)
        )

    def draw_winner_cards(self, frame, found_cards, coords):
        for i in range(len(found_cards)):
            if found_cards[i] in self.winning_cards:
                cv.putText(
                    frame,
                    f"Winner",
                    coords[i],
                    cv.FONT_HERSHEY_COMPLEX,
                    1.0,
                    (255, 255, 0)
                )

        return frame
