import pygame
import math
import random
import keras
import numpy as np
from keras import Model, layers
import pygame
import random
import matplotlib.pyplot as plt
from collections import deque
import sys
from collections import deque

# integrate Replay Buffer mostly everything for better results
memory = deque(maxlen=8000)  
batch_size = 32            

def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


# same as normal QLearning but is better because is learning with large Data = so the model get better context etc.
def replay_experience():
    if len(memory) < batch_size:
        return  
    
    batch = random.sample(memory, batch_size)
    
    states = np.array([e[0] for e in batch])
    actions = np.array([e[1] for e in batch])
    rewards = np.array([e[2] for e in batch])
    next_states = np.array([e[3] for e in batch])
    dones = np.array([e[4] for e in batch])
    
    current_q_values = model.predict(states, verbose=0)
    next_q_values = model.predict(next_states, verbose=0)
    
    for i in range(batch_size):
        if dones[i]:  
            current_q_values[i][actions[i]] = rewards[i]
        else:  
            current_q_values[i][actions[i]] = rewards[i] + 0.95 * np.max(next_q_values[i])
    
    model.fit(states, current_q_values, epochs=1, verbose=0)

model = keras.Sequential((
    keras.Input(shape=(5,)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),          # is good against overfitting 0.2 means that in the Training 20 % of the neueros just get set off
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dense(4, activation="linear")              
))

epsilon = 0.9
epsilon_decay = 0.9995    # epsilon decay for better exploration
epsilon_min = 0.05        

goal_pos = [9, 9]
model.compile(optimizer='adam', loss='mse')

def Qstatelearning(state,action,reward, next_state,alpha=0.1,gamma=0.9):
  state = np.array(state).reshape(1,-1)
  next_state = np.array(next_state).reshape(1,-1)

  q_values = model.predict(state, verbose=0)
  next_q_values = model.predict(next_state, verbose=0)

  target = q_values.copy()
  target[0][action] = reward + gamma * np.max(next_q_values)
  model.fit(state,target,epochs=1,verbose=0)

def getStates(x, y):
    norm_x, norm_y = x / 9.0, y / 9.0
    norm_gx, norm_gy = goal_pos[0] / 9.0, goal_pos[1] / 9.0
    distance = math.dist([x, y], goal_pos) / math.dist([0, 0], [9, 9])
    return np.array([norm_x, norm_y, norm_gx, norm_gy, distance])



# Initialisiere Pygame
pygame.init()

# Einstellungen
WIDTH, HEIGHT = 500, 500
ROWS, COLS = 10, 10
CELL_SIZE = WIDTH // COLS

# Farben
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE  = (0, 0, 255)
GREEN = (0, 255, 0)

# Fenster
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maze Game")

# Labyrinth (0 = frei, 1 = Wand)
maze = [
    [0,0,0,0,0,1,0,0,0,0],
    [1,1,1,1,0,1,0,1,1,0],
    [0,0,0,1,0,0,0,1,0,0],
    [0,1,0,0,0,1,0,1,0,1],
    [0,1,1,1,1,1,0,1,0,1],
    [0,0,0,0,0,0,0,1,0,0],
    [1,1,1,1,1,1,0,1,1,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,1,1,1,0,1,1,1,1,0],
    [0,0,0,1,0,0,0,0,1,0]
]


px = 0
py = 0
def draw():
    global px, py
    for row in range(ROWS):
        for col in range(COLS):
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if maze[row][col] == 1:
                pygame.draw.rect(screen, BLACK, rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)  

    gx, gy = goal_pos
    pygame.draw.rect(screen, GREEN, (gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.draw.rect(screen, BLUE, (px * CELL_SIZE, py * CELL_SIZE, CELL_SIZE, CELL_SIZE))

def checkCollison(x,y):
    if x < 0 or x >= COLS or y < 0 or y >= ROWS:
        return True
    if maze[y][x] == 1:  
        return True
    return False

def checkMazeEnd(x,y):
    gx, gy = goal_pos
    if x == gx and y == gy:
      return True
    else: return False

def move(action):
    global px, py
    dx, dy = 0, 0
    if action == 0: dy = -1   # UP
    if action == 1: dy = 1    # DOWN
    if action == 2: dx = -1   # LEFT
    if action == 3: dx = 1    # RIGHT

    new_x = px + dx
    new_y = py + dy

    if checkCollison(new_x, new_y):  
        return px, py, True  
    return new_x, new_y, False

def getReward(new_x, new_y):
    
    # integrate step counting less steps == more reward
    global px, py
    reward = -1
    collision = checkCollison(new_x,new_y)
    
    if collision:
        reward -= 65
    else:
        reward += 15
    
    if checkMazeEnd(new_x,new_y):
        reward += 200

    gx,gy = goal_pos
    olddist = math.dist([px,py],[gx,gy])
    newdist = math.dist([new_x,new_y],[gx,gy])
    if newdist < olddist:
      reward += 60
    else:
      reward -= 50

    return reward
clock = pygame.time.Clock()
episodes = 0
clock = pygame.time.Clock()
running = True
count = 0
while running:
    screen.fill(WHITE)
    if episodes > 25:
        running = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    state = getStates(px,py)
    if random.random() < epsilon:
        action = random.randint(0, 3)
        count += 1
    else:
        pred = model.predict(state.reshape(1,-1), verbose=0)
        action = np.argmax(pred)
        count += 1
    new_x, new_y, collision = move(action)
    reward = getReward(new_x, new_y)
    if not collision:
      next_state = getStates(new_x, new_y)
      done = ([new_x, new_y] == goal_pos)
      remember(state,action,reward,next_state,done)
      px, py = new_x, new_y
    else:
      next_state = getStates(px, py)
      done = False
      remember(state,action,reward,next_state,done)
    replay_experience()
    if epsilon > epsilon_min:
     epsilon *= epsilon_decay

    if [px, py] == goal_pos:
        print("Won")
        px, py = 0, 0
        episodes += 1
        count = 0
    if count > 500:
        episodes += 1
        px, py = 0, 0
        count = 0
    draw()
    pygame.display.flip()
    clock.tick(60)

if episodes > 25:
  model.save("MazeSolver.keras")


