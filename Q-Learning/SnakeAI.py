import keras
import numpy as np
from keras import  Model, layers
import pygame
import random
import sys


# Prediction Logik auf die Fehler, Q-State learning mit Reward based
# check possible moves, q-state for best move
# move and evaluate
# Position vom Essen




model = keras.Sequential([              
    keras.Input(shape=(11,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(3, activation="linear")
])

model.compile(optimizer='adam', loss='mse')

snake = []
direction_idx = 0
food = (0, 0)
dead = False
screen = None
clock = None

CELL_SIZE = 20
GRID_SIZE = 30
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE

DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

# Farben
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
GREEN = (0, 255, 0)
RED = (255, 0, 0)


def train_q_state_learning(state,action,reward,next_state,alpha=0.1, gamma=0.9):
    state = np.array(state).reshape(1,-1)
    next_state = np.array(next_state).reshape(1,-1)

    q_values = model.predict(state,verbose=0)
    next_q_values = model.predict(next_state,verbose=0)

    target = q_values.copy()
    target[0][action] = reward + gamma * np.max(next_q_values)

    model.fit(state, target, epochs=1, verbose=0)


def get_state():
    head_x, head_y = snake[0]
    direction = DIRECTIONS[direction_idx]

    def check_danger(pos):
        x, y = pos
        return (
            x <= 0 or x >= GRID_SIZE - 1 or
            y <= 0 or y >= GRID_SIZE - 1 or
            (x, y) in snake
        )

    dir_left = (-direction[1], direction[0])
    dir_right = (direction[1], -direction[0])
    dir_straight = direction

    danger_straight = check_danger((head_x + dir_straight[0], head_y + dir_straight[1]))
    danger_right = check_danger((head_x + dir_right[0], head_y + dir_right[1]))
    danger_left = check_danger((head_x + dir_left[0], head_y + dir_left[1]))

    moving_left = direction == (-1, 0)
    moving_right = direction == (1, 0)
    moving_up = direction == (0, -1)
    moving_down = direction == (0, 1)

    food_left = food[0] < head_x
    food_right = food[0] > head_x
    food_up = food[1] < head_y
    food_down = food[1] > head_y

    state = [
        int(danger_straight), int(danger_right), int(danger_left),
        int(moving_left), int(moving_right), int(moving_up), int(moving_down),
        int(food_left), int(food_right), int(food_up), int(food_down)
    ]

    return np.array(state)

    
  
CELL_SIZE = 20
GRID_SIZE = 30
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE

DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
GREEN = (0, 255, 0)
RED = (255, 0, 0)


def init():
    global screen, clock
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    reset()

def reset():
    global snake, direction_idx, food, dead
    snake = [(5, 5), (5, 6), (5, 7)]
    direction_idx = 0
    food = place_food()
    dead = False

def place_food():
    while True:
        pos = (random.randint(1, GRID_SIZE - 2), random.randint(1, GRID_SIZE - 2))
        if pos not in snake:
            return pos

def move(action):
    
    global direction_idx, snake, food, dead

    if dead:
        return -10
    
    old_distance = abs(snake[0][0] - food[0]) + abs(snake[0][1] - food[1])

    if action == 1:
        direction_idx = (direction_idx + 1) % 4
    elif action == 2:
        direction_idx = (direction_idx - 1) % 4

    direction = DIRECTIONS[direction_idx]
    head = snake[0]
    new_head = (head[0] + direction[0], head[1] + direction[1])

    if new_head[0] <= 0 or new_head[0] >= GRID_SIZE - 1 or new_head[1] <= 0 or new_head[1] >= GRID_SIZE - 1:
        dead = True
        return -10

    if new_head in snake:
        dead = True
        return -10

    snake.insert(0, new_head)

    if new_head == food:
        food = place_food()
        return 20
    else:
        snake.pop()
        new_distance = abs(new_head[0] - food[0]) + abs(new_head[1] - food[1])
        if new_distance < old_distance:
            return 1  
        else:
            return -1  

def render():
    # Pygame Events verarbeiten
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    screen.fill(BLACK)

    # Wände zeichnen
    for i in range(GRID_SIZE):
        pygame.draw.rect(screen, GRAY, (i * CELL_SIZE, 0, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(screen, GRAY, (i * CELL_SIZE, (GRID_SIZE - 1) * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(screen, GRAY, (0, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(screen, GRAY, ((GRID_SIZE - 1) * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Schlange zeichnen
    for segment in snake:
        pygame.draw.rect(screen, GREEN, (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Futter zeichnen
    pygame.draw.rect(screen, RED, (food[0] * CELL_SIZE, food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()
    clock.tick(10)

init()

for episode in range(1000):
    reset()
    while not dead:
        state = get_state()
        if random.random() < 0.1:  # epsilon-greedy
            action = random.randint(0, 2)
        else:
            prediction = model.predict(state.reshape(1, -1), verbose=0)
            action = np.argmax(prediction)
        reward = move(action)
        next_state = get_state()            # Performance Optimization wenn zu lange kein Futter minus Reward
        train_q_state_learning(state, action, reward, next_state)
        render()
      
model.save("Snake.keras")


pygame.quit()