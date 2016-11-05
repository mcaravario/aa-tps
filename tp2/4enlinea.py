import random
from itertools import groupby, chain
import matplotlib.pyplot as plt
from sklearn.grid_search import ParameterGrid

def diagonalsPos (matrix, cols, rows):
    """Get positive diagonals, going from bottom-left to top-right."""
    for di in ([(j, i - j) for j in range(cols)] for i in range(cols + rows -1)):
        yield [matrix[i][j] for i, j in di if i >= 0 and j >= 0 and i < cols and j < rows]

def diagonalsNeg (matrix, cols, rows):
    """Get negative diagonals, going from top-left to bottom-right."""
    for di in ([(j, i - cols + j + 1) for j in range(cols)] for i in range(cols + rows - 1)):
        yield [matrix[i][j] for i, j in di if i >= 0 and j >= 0 and i < cols and j < rows]

NONE = "."

class CuatroEnLinea:
    def __init__(self, playerX, playerO, cols = 3, rows = 3, requiredToWin = 3):
        """Create a new game."""
        self.cols = cols
        self.rows = rows
        self.win = requiredToWin
        self.board = [[NONE] * rows for _ in range(cols)]
        self.playerX, self.playerO = playerX, playerO
        self.playerX_turn = random.choice([True, False])


    def play_game(self):
        self.playerX.start_game('X', self.cols, self.rows)
        self.playerO.start_game('O', self.cols, self.rows)
        while True: #yolo
            if self.playerX_turn:
                player, char, other_player = self.playerX, 'X', self.playerO
            else:
                player, char, other_player = self.playerO, 'O', self.playerX
            if player.breed == "human":
                self.display_board()
            col = player.move(self.board)

            self.insert(col, char)

            if self.player_wins(char):
                player.reward(1, self.board)
                other_player.reward(-1, self.board)
                return player

                break
            if self.board_full(): # tie game
                player.reward(0.5, self.board)
                other_player.reward(0.5, self.board)   
                return None
                break
            other_player.reward(0, self.board)
            self.playerX_turn = not self.playerX_turn

    def insert (self, column, char):
        """Insert the char in the given column."""
        if column > self.cols-1:
            raise Exception('Invalid move')
        c = self.board[column]
        if c[0] != NONE:
            raise Exception('Column is full')

        i = -1
        while c[i] != NONE:
            i -= 1
        c[i] = char
    
    def player_wins(self, player_char):
        """Get the winner on the current board."""
        lines = (
            self.board, # columns
            zip(*self.board), # rows
            diagonalsPos(self.board, self.cols, self.rows), # positive diagonals
            diagonalsNeg(self.board, self.cols, self.rows) # negative diagonals
        )

        for line in chain(*lines):
            for char, group in groupby(line):
                if char == player_char and len(list(group)) >= self.win:
                    return True
        return False        

    def board_full(self):
        return not any([space == NONE for cols in self.board for space in cols])

    def display_board(self):
        """Print the board."""
        print "  ", '  '.join(map(str, range(self.cols)))
        for y in range(self.rows):
            print str(y)+" ", '  '.join(str(self.board[x][y]) for x in range(self.cols))
        print "\n"

class Player(object):
    def __init__(self,):
        self.breed = "human"

    def start_game(self, char, cols, row):
        print "\nNew game!"

    def move(self, board):
        return int(raw_input("Your move? "))

    def reward(self, value, board):
        print "{} rewarded: {}".format(self.breed, value)

    def available_moves(self, board):
        return [i for i in xrange(len(board)) if board[i][0] == NONE]


class RandomPlayer(Player):
    def __init__(self):
        self.breed = "random"

    def reward(self, value, board):
        pass

    def start_game(self, char, cols, rows):
        pass

    def move(self, board):
        return random.choice(self.available_moves(board))

class QLearningPlayer(Player):
    def __init__(self, epsilon=0.2, learning_rate=0.3, discount=0.9):
        self.breed = "Qlearner"
        self.harm_humans = False
        self.q = {} # (state, action) keys: Q values
        self.epsilon = epsilon # e-greedy chance of random exploration
        self.learning_rate = learning_rate # learning rate
        self.discount = discount # discount factor for future rewards

    def start_game(self, char, cols, rows):
        self.last_board = [[NONE] * rows for _ in range(cols)]
        self.last_move = None

    def move(self, board):
        self.last_board = tuple([tuple(col) for col in board])
        actions = self.available_moves(board)

        if random.random() < self.epsilon: # explore!
            self.last_move = random.choice(actions)
            return self.last_move

        qs = [self.getQ(self.last_board, a) for a in actions]
        maxQ = max(qs)

        if qs.count(maxQ) > 1:
            # more than 1 best option; choose among them randomly
            best_options = [i for i in range(len(actions)) if qs[i] == maxQ]
            i = random.choice(best_options)
        else:
            i = qs.index(maxQ)

        self.last_move = actions[i]
        return actions[i]

    def reward(self, value, board):
        if self.last_move:
            self.learn(self.last_board, self.last_move, value, tuple([tuple(col) for col in board]))

    def getQ(self, state, action):
        # encourage exploration; "optimistic" 1.0 initial values
        if self.q.get((state, action)) is None:
            self.q[(state, action)] = 1.0
        return self.q.get((state, action))

    def learn(self, state, action, reward, result_state):
        prev = self.getQ(state, action)
        maxqnew = max([self.getQ(result_state, a) for a in self.available_moves(state)])
        self.q[(state, action)] = prev + self.learning_rate * ((reward + self.discount*maxqnew) - prev)

def gridSeach(param_grid):
    grid = ParameterGrid(param_grid)
    for params in grid:
        p1 = QLearningPlayer(params["epsilon"], params["learning_rate"], params["discount"])
        p2 = RandomPlayer()

        p1_wins = 0

        for i in xrange(1,500000):
            t = CuatroEnLinea(p1, p2)
            winner = t.play_game()
            if winner == p1:
                p1_wins += 1

            if p1_wins / float(i) > 0.7:
                return params
    return None

# [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], "learning_rate": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], "discount": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
def experis():
    param_grid = {"epsilon": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], "learning_rate": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], "discount": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
    best_params = gridSeach(param_grid)
    if best_params:
        print best_params
    print "No se encontraron buenos parametros"

experis()

# p1 = QLearningPlayer()
# p2 = RandomPlayer()

# p1_wins = [0]
# p2_wins = [0]

# for i in xrange(0,500000):
#     t = CuatroEnLinea(p1, p2)
#     winner = t.play_game()
#     if winner == p1:
#         p1_wins.append(p1_wins[-1]+1)
#         p2_wins.append(p2_wins[-1])
#     elif winner == p2:
#         p2_wins.append(p2_wins[-1]+1)
#         p1_wins.append(p1_wins[-1])
#     else:
#         p1_wins.append(p1_wins[-1])        
#         p2_wins.append(p2_wins[-1])

# for i in xrange(1, 500001):
#     p1_wins[i] = p1_wins[i]/float(i)
#     p2_wins[i] = p2_wins[i]/float(i)

# plt.plot(p1_wins)
# plt.plot(p2_wins)
# plt.show()

# p1 = Player()
# p2.epsilon = 0
# # p2 = Player()

# # while True:
# #     t = CuatroEnLinea(p1, p2)
# #     t.play_game()
