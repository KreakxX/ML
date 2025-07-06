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

# Pygame initialisieren
pygame.init()

# Konstanten
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
FPS = 60

# Farben
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)

memory = deque(maxlen=2000)  
batch_size = 32            
train_counter = 0  

# Auto-Variablen
car_x = SCREEN_WIDTH // 2
car_y = SCREEN_HEIGHT // 2 + 200
car_angle = 0
car_speed = 4 # Konstante Geschwindigkeit
car_turn_speed = 3
car_width = 20
car_height = 10
car_alive = True
car_distance_traveled = 0
car_last_x = car_x
car_last_y = car_y
car_reward = 0
epsilon = 0.9 
min_epsilon = 0.05 
decay_rate = 0.9995

outer_points = []
inner_points = []
action_index = 0

# Reset-Timer
reset_timer = 0
reset_delay = 3

# Neural Network für nur 3 Aktionen: links, geradeaus, rechts
model = keras.Sequential([              
    keras.Input(shape=(10,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(3, activation="linear")  # 3 Aktionen: links, geradeaus, rechts
])

model.compile(optimizer='adam', loss='mse')


# Replay Buffer

def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def replay_experience():
    if len(memory) < batch_size:
        return  # Noch nicht genug Erfahrungen
    
    # Zufällige Auswahl von Erfahrungen
    batch = random.sample(memory, batch_size)
    
    # Daten aufteilen
    states = np.array([e[0] for e in batch])
    actions = np.array([e[1] for e in batch])
    rewards = np.array([e[2] for e in batch])
    next_states = np.array([e[3] for e in batch])
    dones = np.array([e[4] for e in batch])
    
    # Q-Werte berechnen
    current_q_values = model.predict(states, verbose=0)
    next_q_values = model.predict(next_states, verbose=0)
    
    # Ziel-Q-Werte aktualisieren
    for i in range(batch_size):
        if dones[i]:  # Spiel beendet (Crash)
            current_q_values[i][actions[i]] = rewards[i]
        else:  # Spiel läuft weiter
            current_q_values[i][actions[i]] = rewards[i] + 0.95 * np.max(next_q_values[i])
    
    # Netzwerk trainieren
    model.fit(states, current_q_values, epochs=1, verbose=0)

# def train_q_state_learning(state, action, reward, next_state, alpha=0.01, gamma=0.95):    
#     state = np.array(state).reshape(1, -1)
#     next_state = np.array(next_state).reshape(1, -1)

#     q_values = model.predict(state, verbose=0)
#     next_q_values = model.predict(next_state, verbose=0)

#     target = q_values.copy()
#     target[0][action] = reward + gamma * np.max(next_q_values)
#     model.fit(state, target, epochs=1, verbose=0)

def cast_ray(x, y, angle_deg, max_length=300):
    rad = math.radians(angle_deg)
    for dist in range(0, max_length, 5):
        check_x = x + math.cos(rad) * dist
        check_y = y + math.sin(rad) * dist
        if not point_on_track((check_x, check_y), outer_points + inner_points):
            return dist
    return max_length

def calculate_reward():
    # Bewegungsbelohnung
    dx = car_x - car_last_x
    dy = car_y - car_last_y
    distance_moved = math.sqrt(dx*dx + dy*dy)
    movement_reward = distance_moved * 0.5
    
    # Sensor-basierte Belohnung
    front, back, left, right = getDistancesFromCar()
    
    # Belohnung für Balance zwischen links und rechts
    if left + right > 0:
        balance_reward = 1.0 - abs(left - right) / (left + right)
    else:
        balance_reward = 0
    
    # Belohnung für Vorwärtsfahrt
    forward_reward = min(front / 100.0, 1.0)
    
    total_reward = movement_reward + balance_reward * 0.5 + forward_reward * 0.3
    return total_reward

def getDistancesFromCar():
    angles = {
        "front": car_angle,
        "back": (car_angle + 180) % 360,
        "left": (car_angle - 90) % 360,
        "right": (car_angle + 90) % 360,
    }
    
    distance_front = cast_ray(car_x, car_y, angles["front"])
    distance_back = cast_ray(car_x, car_y, angles["back"])          
    distance_left = cast_ray(car_x, car_y, angles["left"])
    distance_right = cast_ray(car_x, car_y, angles["right"])
    
    return distance_front, distance_back, distance_left, distance_right

def getStates():
    global outer_points, inner_points                   
    
    point = [car_x, car_y]
    danger_nextMove = not point_on_track(point, outer_points + inner_points)
    distance_left, distance_right, distance_front, distance_back = getDistancesFromCar()
    max_distance = math.sqrt(SCREEN_WIDTH*SCREEN_WIDTH + SCREEN_HEIGHT*SCREEN_HEIGHT)
    state = [
            (car_x - SCREEN_WIDTH/2) / (SCREEN_WIDTH/2),       
            (car_y - SCREEN_HEIGHT/2) / (SCREEN_HEIGHT/2),     
            (car_angle % 360) / 180.0 - 1.0,                    
            car_speed / 5.0,                     
            int(car_alive),                                     
            int(danger_nextMove),                                 
            distance_front / max_distance,                      
            distance_back / max_distance,                           
            distance_left / max_distance,                                
            distance_right / max_distance 
        ]
    return np.array(state)

def generate_track():
    points = []
    center_x, center_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
    
    for i in range(0, 360, 10):
        angle = math.radians(i)
        if 0 <= i < 90:
            radius = 200
        elif 90 <= i < 180:
            radius = 250
        elif 180 <= i < 270:
            radius = 180
        else:
            radius = 220
            
        x = center_x + math.cos(angle) * radius
        y = center_y + math.sin(angle) * radius * 0.7
        points.append((x, y))
    
    return points

def update_car(action):
    global car_x, car_y, car_angle, car_last_x, car_last_y, car_distance_traveled, car_alive, car_reward
    
    if not car_alive:
        return
    
    # Nur Lenkung:
    # action 0: links lenken
    # action 1: geradeaus fahren
    # action 2: rechts lenken
    
    if action == 0:  # Links lenken
        car_angle -= car_turn_speed
    elif action == 2:  # Rechts lenken
        car_angle += car_turn_speed
    # action == 1: geradeaus (nichts tun)
    
    car_last_x = car_x
    car_last_y = car_y
    
    # Auto fährt immer mit konstanter Geschwindigkeit vorwärts
    car_x += math.cos(math.radians(car_angle)) * car_speed
    car_y += math.sin(math.radians(car_angle)) * car_speed
    
    dx = car_x - car_last_x
    dy = car_y - car_last_y
    distance_this_frame = math.sqrt(dx*dx + dy*dy)
    car_distance_traveled += distance_this_frame
    
    # Grundbelohnung für Bewegung
    car_reward += distance_this_frame * 0.1

    return car_distance_traveled

def get_car_corners():
    cos_angle = math.cos(math.radians(car_angle))
    sin_angle = math.sin(math.radians(car_angle))
    
    corners = [
        (-car_width/2, -car_height/2),
        (car_width/2, -car_height/2),
        (car_width/2, car_height/2),
        (-car_width/2, car_height/2)
    ]

    rotated_corners = []
    for corner_x, corner_y in corners:
        rotated_x = corner_x * cos_angle - corner_y * sin_angle
        rotated_y = corner_x * sin_angle + corner_y * cos_angle
        rotated_corners.append((car_x + rotated_x, car_y + rotated_y))
    
    return rotated_corners

def draw_car(screen):
    if not car_alive:
        return
        
    corners = get_car_corners()
    pygame.draw.polygon(screen, RED, corners)
    
    # Richtung anzeigen
    front_x = car_x + math.cos(math.radians(car_angle)) * car_width/2
    front_y = car_y + math.sin(math.radians(car_angle)) * car_width/2
    pygame.draw.circle(screen, YELLOW, (int(front_x), int(front_y)), 3)

def draw_track(screen, track_points):
    global outer_points, inner_points
    track_width = 40
    outer_points = []
    inner_points = []
    
    for i, point in enumerate(track_points):
        next_point = track_points[(i + 1) % len(track_points)]
        
        dx = next_point[0] - point[0]
        dy = next_point[1] - point[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            normal_x = -dy / length
            normal_y = dx / length
            
            outer_x = point[0] + normal_x * track_width/2
            outer_y = point[1] + normal_y * track_width/2
            inner_x = point[0] - normal_x * track_width/2
            inner_y = point[1] - normal_y * track_width/2
            
            outer_points.append((outer_x, outer_y))
            inner_points.append((inner_x, inner_y))
    
    pygame.draw.polygon(screen, GRAY, outer_points)
    pygame.draw.polygon(screen, DARK_GRAY, inner_points)
    
    pygame.draw.lines(screen, WHITE, True, track_points, 40)

def point_on_track(point, track_points):
    x, y = point
    track_width = 40
    
    min_distance = float('inf')
    for track_point in track_points:
        dx = x - track_point[0]
        dy = y - track_point[1]
        distance = math.sqrt(dx*dx + dy*dy)
        min_distance = min(min_distance, distance)
    
    return min_distance <= track_width/2

def check_collision(track_points, current_action, state):
    global car_alive, car_reward, reset_timer

    car_corners = get_car_corners()
    
    for corner in car_corners:
        if not point_on_track(corner, track_points):
            car_alive = False
            car_reward = -50  # Feste Strafe für Crash
            reset_timer = reset_delay
            return True
    return False

def reset_game():
    global car_x, car_y, car_angle, car_alive, car_distance_traveled
    global car_last_x, car_last_y, car_reward, reset_timer
    
    track_points = generate_track()
    spawn_point = track_points[0]   
    car_x = spawn_point[0]
    car_y = spawn_point[1]
    
    if len(track_points) > 1:
        next_point = track_points[1]
        dx = next_point[0] - car_x
        dy = next_point[1] - car_y
        car_angle = math.degrees(math.atan2(dy, dx))
    else:
        car_angle = 0

    car_alive = True
    car_distance_traveled = 0
    car_last_x = car_x
    car_last_y = car_y
    car_reward = 0
    reset_timer = 0

def main():
    global car_reward, reset_timer, epsilon, action_index,train_counter
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("2D Auto Rennspiel - Nur Lenkung lernen")
    clock = pygame.time.Clock()
    
    track_points = generate_track()
    
    reset_game()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset_game()
        
        if reset_timer > 0:                    
            reset_timer -= 1
            if reset_timer == 0:
                reset_game()
        
        if car_alive:
            state = getStates()

            # Entscheidung: Zufällig oder KI-gesteuert
            if random.random() < epsilon:  
                action_index = random.randint(0, 2)  # 0=links, 1=geradeaus, 2=rechts
            else:
                raw_action = model.predict(state.reshape(1, -1), verbose=0)[0]
                action_index = np.argmax(raw_action)

            # Auto updaten
            update_car(action_index)
            
            sensor_reward = calculate_reward()
            car_reward += sensor_reward

            # Kollision prüfen
            next_state = getStates()
            crashed = check_collision(track_points, action_index, state)

            if crashed:
                # Crash-Erfahrung speichern
                remember(state, action_index, car_reward, next_state, True)
            else:
                # Normale Erfahrung speichern
                remember(state, action_index, car_reward, next_state, False)

            train_counter += 1
            if train_counter % 4 == 0 and len(memory) > batch_size:
                replay_experience()

            car_reward = 0  
                        
        epsilon = max(min_epsilon, epsilon * decay_rate)

        # Zeichnen
        screen.fill(BLACK)
        draw_track(screen, track_points)
        draw_car(screen)
        
        # Debug-Info
        font = pygame.font.Font(None, 36)
        epsilon_text = font.render(f"Epsilon: {epsilon:.3f}", True, WHITE)
        screen.blit(epsilon_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()

if __name__ == "__main__":
    main()