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

    game_state = 0

    #TRUMP_SUITS = ["Spades", "Hearts", "Clubs", "Diamonds"] #Copas, Espadas, Ouros, Paus


    def __init__(self, number_rounds):
        self.vertical_team_points = 0
        self.horizontal_team_points = 0
        self.cards_played = 0
        self.total_number_rounds = int(number_rounds)
        
    def is_game_over(self):
        return self.total_number_rounds <= self.cards_played / 4
    
    def setTrumpSuit(self, trump_suit):
        self.trump_suit = trump_suit
        self.game_state = 1
        print("set trump suit: " + trump_suit)

    def getWinner(self):
        if(not self.is_game_over):
            return -1
        if(self.team_1_points > self.team_2_points):
            return 1
        elif(self.team_2_points > self.team_1_points):
            return 2
        return 0
        #if(self.team_1_points > 60):
        #    return 1
        #if(self.team_2_points > 60):
        #    return 2

    def calcRound(self, card1Suit, card1Value, card2Suit, card2Value, card3Suit, card3Value, card4Suit, card4Value):
        if(self.CARD_TYPE_TO_POINTS[card1Value] < 0 or self.CARD_TYPE_TO_POINTS[card2Value] < 0 or self.CARD_TYPE_TO_POINTS[card3Value] < 0 or self.CARD_TYPE_TO_POINTS[card4Value] < 0):
            return False
        
        # Calculate winner
        card1Prio = (999 if card1Suit == self.trump_suit else 0) + self.CARD_TYPE_PRIORITY[card1Value]
        card2Prio = (999 if card2Suit == self.trump_suit else 0) + self.CARD_TYPE_PRIORITY[card2Value]
        card3Prio = (999 if card3Suit == self.trump_suit else 0) + self.CARD_TYPE_PRIORITY[card3Value]
        card4Prio = (999 if card4Suit == self.trump_suit else 0) + self.CARD_TYPE_PRIORITY[card4Value]

        max_prio = max([card1Prio, card2Prio, card3Prio, card4Prio])

        card1Points = self.CARD_TYPE_TO_POINTS[card1Value]
        card2Points = self.CARD_TYPE_TO_POINTS[card2Value]
        card3Points = self.CARD_TYPE_TO_POINTS[card3Value]
        card4Points = self.CARD_TYPE_TO_POINTS[card4Value]

        roundPoints = card1Points + card2Points + card3Points + card4Points

        # Add points to winner's team
        if(card1Prio == max_prio or card3Prio == max_prio):
            self.team_1_points += roundPoints
        if(card2Prio == max_prio or card4Prio == max_prio):
            self.team_2_points += roundPoints

        self.cards_played += 4
        if(self.cards_played >= 4 * self.total_number_rounds):
            self.is_game_over = True
            self.game_state = 2
        
    def tryAddCards(self, list_suits, list_values):
        for i in range(0, len(list_suits)):
            already_in_list = False
            for j in range(0, len(self.play_cards_suits)):
                if(list_suits[i] == self.play_cards_suits[j] and list_values[i] == self.play_cards_values[j]):
                    already_in_list = True
            if(not already_in_list):
                (self.play_cards_suits).append(list_suits[i])
                (self.play_cards_values).append(list_values[i])
        if (len(self.play_cards_suits) >= 4):
            self.calcRound(self.play_cards_suits[0], self.play_cards_values[0], self.play_cards_suits[1], self.play_cards_values[1], self.play_cards_suits[2], self.play_cards_values[2], self.play_cards_suits[3], self.play_cards_values[3])
            self.play_cards_suits.clear()
            self.play_cards_values.clear()
