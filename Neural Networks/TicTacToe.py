
import keras
import numpy as np
from keras import Input, Model, layers
import copy
import random

spielFeld = [[0 for _ in range(3)] for _ in range(3)]
model = keras.Sequential([              # basic neural network
    keras.Input(shape=(9,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(9, activation="softmax")
])

model.compile(optimizer='adam', loss='mse')
replay_buffer = []


def makeAMove(x, y, c):         # function for AI
    if spielFeld[x][y] == 0:
        spielFeld[x][y] = c
        return True
    return False


def randomMove():       # function for opponent of AI
    empty_cells = [(i, j) for i in range(3) for j in range(3) if spielFeld[i][j] == 0]
    if not empty_cells:
        return None

    return random.choice(empty_cells)

def checkWin():     # function for checking if the AI or Opponent won
    for row in spielFeld:
        if row[0] == row[1] == row[2] != 0:
            return row[0]

    for col in range(3):
        if spielFeld[0][col] == spielFeld[1][col] == spielFeld[2][col] != 0:
            return spielFeld[0][col]

    if spielFeld[0][0] == spielFeld[1][1] == spielFeld[2][2] != 0:
        return spielFeld[0][0]
    if spielFeld[0][2] == spielFeld[1][1] == spielFeld[2][0] != 0:
        return spielFeld[0][2]

    return 0


def getState():
    return np.array([spielFeld[i][j] for i in range(3) for j in range(3)])


def getReward(winner, moves, possiblechances): # funktion damit die AI guckt ob Gewinnt Weiterbildung
    if winner == 1 and moves <= 5:
        return 2                     # reward big if a fast win
    if winner == 1 and moves <= 10:
        return 1.6          # reward mid if a normal win        # noch mehr bestrafung
    elif winner == 1:
        return 1.3          # reward normal if its a slow win
    elif winner == -1:
        return -1.1 * possiblechances  # if he couldve won but lost than less reward
    return 0.1


def SelectAction(state, epsilon=0.1):
    empty_cells = [(i, j) for i in range(3) for j in range(3) if spielFeld[i][
        j] == 0]  # get all the empty cells (possible moves) by iterating and checking if the position ther e is 0

    if random.uniform(0, 1) < epsilon:      # Exploration a Random move, based on the possible ones
        return random.choice(empty_cells)
    else:   # best move (Exploitation)
        state = np.array(state).reshape(1, -1)      # Implement checking if position is possible or not, if this is case continue else break
        q_values = model.predict(state)[0]

        q_values_masked = np.full_like(q_values ,-np.inf)       # makes minus infinity to the q_values
        for (i, j) in empty_cells:
            idx = i * 3 + j
            q_values_masked[idx] = q_values[idx]        # only takes moves that are possible
        best_action = np.argmax(q_values_masked)        # and takes the highest of the possible moves and plays it eventually
        return (best_action // 3, best_action % 3)


def trainQLearning(state,action,reward,next_state,alpha=0.1, gamma=0.9):
    state = np.array(state).reshape(1, -1)
    next_state = np.array(next_state).reshape(1,-1)

    q_values = model.predict(state, verbose=0)[0]   #logs verbose
    next_q_values = model.predict(next_state, verbose=0)[0]

    index = action[0] * 3 + action[1]

    q_values[index] = q_values[index] + alpha * (reward + gamma * np.max(next_q_values) - q_values[index]) # this is the Q learning Formula Q(s, a) ← Q(s, a) + α * [reward + γ * max(Q(s’, a’)) - Q(s, a)]

    q_values = np.reshape(q_values, (1, 9))     # reshape for reshaping an array

    model.fit(state, q_values, epochs=1, verbose=0, batch_size=64)

def checkForPossibleWin():
    possible_moves = [(i, j) for i in range(3) for j in range(3) if spielFeld[i][j] == 0]

    for move in possible_moves:
        simulated_field = copy.deepcopy(spielFeld)
        simulated_field[move[0]][move[1]] = 1

        if checkWinSimulated(simulated_field) == 1:
            return 1

    return 0

def checkWinSimulated(field):
    for i in range(3):
        if field[i][0] == field[i][1] == field[i][2] != 0:
            return field[i][0]
        if field[0][i] == field[1][i] == field[2][i] != 0:
            return field[0][i]
    if field[0][0] == field[1][1] == field[2][2] != 0:
        return field[0][0]
    if field[0][2] == field[1][1] == field[2][0] != 0:
        return field[0][2]
    return -1

def playGame(epsilon):             # Ai against himself playing
    winner = 0
    turn = 1
    moves = 0

    while winner == 0:
        if turn == 1:
            state = getState()
            action = SelectAction(state, epsilon)
            if not makeAMove(action[0], action[1],1):
                break
            winner = checkWin()
            a = checkForPossibleWin()
            reward = getReward(winner, moves, a)
            next_state = getState()
            replay_buffer.append((state, action, reward,next_state))
            trainQLearning(state, action, reward, next_state)
            if winner != 0:
                break
        else:
            action = randomMove()
            if action is None:
                break
            makeAMove(action[0], action[1], -1)
            winner = checkWin()

        turn = -turn
        moves += 1
        if moves >= 9:
            winner = 0
            break


    return winner

def StartGame():
    global spielFeld
    global replay_buffer
    epsilon = 1.0

    for episode in range(700):
        spielFeld = [[0 for _ in range(3)] for _ in range(3)]
        winner = playGame(epsilon)
        epsilon = max(0.01, epsilon * 0.995)
        print(f"Episode {episode} beendet. Gewinner: {winner}")
        for sample in random.sample(replay_buffer, min(20, len(replay_buffer))):
            trainQLearning(*sample)
    model.save("TicTacToe.keras")


StartGame()


def playAgainstAI():
    replay_buffer_new = []
    ai_transitions = []
    model = keras.models.load_model("TicTacToe.keras")
    global spielFeld
    spielFeld = [[0 for _ in range(3)] for _ in range(3)]

    def printBoard():
        for row in spielFeld:
            print(" | ".join(str(cell) if cell != 0 else " " for cell in row))
            print("-" * 9)

    printBoard()
    turn = 1
    moves = 0
    reward = 0
    while True:
        printBoard()
        if checkWin() == 1:
            print("Ki hat gewonnen")
            reward = getReward(1,moves,checkForPossibleWin())
            break
        elif checkWin() == -1:
            print("Ich habe gewonne")
            reward = getReward(-1,moves,checkForPossibleWin())

            break
        elif all(cell != 0 for row in spielFeld for cell in row):
            print("Unentschieden!")
            reward = getReward(0,moves,checkForPossibleWin())

            break

        if turn == 1:
            state = getState()
            action = SelectAction(state, epsilon= 0)
            makeAMove(action[0],action[1],1)

            next_state = getState()
            ai_transitions.append((state, action, next_state))

        else:
            try:
                user_input = input("Dein Zug:")
                x,y = map(int, user_input.strip().split())
                if not makeAMove(x,y,-1):
                    print("Already set")
                    continue
            except:
                print("Error")

        turn *= -1
        moves += 1

    for state, action, next_state in ai_transitions:
        replay_buffer_new.append((state, action, reward, next_state))

    for state, action, reward, next_state in replay_buffer_new:
        trainQLearning(state, action, reward, next_state)

    model.save("TicTacToe.keras")

#playAgainstAI()

# implement AI playing against self
