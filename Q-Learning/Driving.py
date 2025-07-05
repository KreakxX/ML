import pygame
import math
import random
import keras
import numpy as np
from keras import  Model, layers
import pygame
import random

import matplotlib.pyplot as plt

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

# Auto-Variablen
car_x = SCREEN_WIDTH // 2
car_y = SCREEN_HEIGHT // 2 + 200
car_angle = 0
car_speed = 0
car_max_speed = 5
car_acceleration = 0.2
car_friction = 0.1
car_turn_speed = 3
car_width = 20
car_height = 10
car_alive = True
car_distance_traveled = 0
car_last_x = car_x
car_last_y = car_y
car_checkpoints_passed = 0
car_reward = 0
epsilon = 1.0        # Anfangs hohe Zufallsrate (100 % Zufall)
min_epsilon = 0.05   # Minimal erlaubte Zufallsrate (5 % Zufall)
decay_rate = 0.9995  # Langsamerer Decay

outer_points = []
inner_points = []
action_index = 0

# Reset-Timer
reset_timer = 0
reset_delay = 120  

model = keras.Sequential([              
    keras.Input(shape=(9,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(4, activation="linear")
])


model.compile(optimizer='adam', loss='mse')
state = [
    car_x,           
    car_y,             
    car_angle,       
    car_speed,       
    car_reward,      
    car_alive       
]

def normalize_state(state):
    """Normalisiert die Zustandswerte für besseres neuronales Netzwerk-Training"""
    normalized = np.array(state, dtype=np.float32)
    
    normalized[0] = (normalized[0] - SCREEN_WIDTH/2) / (SCREEN_WIDTH/2)   # car_x
    normalized[1] = (normalized[1] - SCREEN_HEIGHT/2) / (SCREEN_HEIGHT/2) # car_y
    
    normalized[2] = (normalized[2] % 360) / 180.0 - 1.0  # car_angle
    
    normalized[3] = normalized[3] / car_max_speed  # car_speed
    
    
    return normalized

def train_q_state_learning(state, action, reward, next_state, alpha=0.1, gamma=0.9):
    # Zustand normalisieren vor dem Training
    normalized_state = normalize_state(state)
    normalized_next_state = normalize_state(next_state)
    
    normalized_state = normalized_state.reshape(1, -1)
    normalized_next_state = normalized_next_state.reshape(1, -1)

    q_values = model.predict(normalized_state, verbose=0)
    next_q_values = model.predict(normalized_next_state, verbose=0)

    target = q_values.copy()
    target[0][action] = reward + gamma * np.max(next_q_values)
    model.fit(normalized_state, target, epochs=1, verbose=0)

def danger_ahead():
    forward_distance = 50
    back_distance = 50
    front_x = car_x + math.cos(math.radians(car_angle)) * forward_distance
    front_y = car_y + math.sin(math.radians(car_angle)) * forward_distance
    back_x = car_x - math.cos(math.radians(car_angle)) * back_distance
    back_y = car_y - math.sin(math.radians(car_angle)) * back_distance
    front_off_track = not point_on_track((front_x, front_y), outer_points + inner_points)
    back_off_track = not point_on_track((back_x, back_y), outer_points + inner_points)

    return front_off_track, back_off_track


    
def getStates():
    global outer_points, inner_points, car_reward
    point= [car_x,car_y]                                            # more states #
    danger_nextMove = point_on_track(point, outer_points + inner_points)
    danger_front, danger_back = danger_ahead()

    state = [
        car_x,           
        car_y,             
        car_angle,       
        car_speed,       
        car_alive,
        int(car_alive),
        int(danger_nextMove),
        int(danger_front),
        int(danger_back)
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
        y = center_y + math.sin(angle) * radius * 0.7  # Oval-Form
        points.append((x, y))
    
    return points

# Checkpoints generieren
def generate_checkpoints(track_points):
    checkpoints = []
    for i in range(0, len(track_points), len(track_points) // 8):
        checkpoints.append(track_points[i])
    return checkpoints

# Auto-Funktionen
def update_car(action):
    global car_x, car_y, car_angle, car_speed, car_last_x, car_last_y, car_distance_traveled, car_alive, car_reward
    
    if not car_alive:
        return
    
    # Beschleunigung/Bremsen
    if action[0]:  # Beschleunigen
        car_speed = min(car_speed + car_acceleration, car_max_speed)
    elif action[1]:  # Bremsen
        car_speed = max(car_speed - car_acceleration * 2, -car_max_speed * 0.5)
    else:
        # Reibung anwenden
        if car_speed > 0:
            car_speed = max(car_speed - car_friction, 0)
        elif car_speed < 0:
            car_speed = min(car_speed + car_friction, 0)
    
    # Lenken (nur wenn sich das Auto bewegt)
    if abs(car_speed) > 0.1:
        if action[2]:  # Links
            car_angle -= car_turn_speed
        if action[3]:  # Rechts
            car_angle += car_turn_speed
    
    # Position aktualisieren
    ontrack = point_on_track([car_x, car_y],outer_points + inner_points)

    if ontrack:
        car_reward += 20
    else:
        car_reward -= 10

    # only when moving on track than give reward



    car_last_x = car_x
    car_last_y = car_y
    
    car_x += math.cos(math.radians(car_angle)) * car_speed
    car_y += math.sin(math.radians(car_angle)) * car_speed
    
    # Distanz berechnen
    dx = car_x - car_last_x
    dy = car_y - car_last_y
    car_distance_traveled += math.sqrt(dx*dx + dy*dy)

def get_car_corners():
    cos_angle = math.cos(math.radians(car_angle))
    sin_angle = math.sin(math.radians(car_angle))
    
    # Relative Positionen der Ecken
    corners = [
        (-car_width/2, -car_height/2),
        (car_width/2, -car_height/2),
        (car_width/2, car_height/2),
        (-car_width/2, car_height/2)
    ]
    
    # Rotieren und positionieren
    rotated_corners = []
    for corner_x, corner_y in corners:
        rotated_x = corner_x * cos_angle - corner_y * sin_angle
        rotated_y = corner_x * sin_angle + corner_y * cos_angle
        rotated_corners.append((car_x + rotated_x, car_y + rotated_y))
    
    return rotated_corners
def distance_to_next_checkpoint(checkpoints):
    if len(checkpoints) == 0:
        return float('inf')
    
    current_checkpoint = checkpoints[car_checkpoints_passed % len(checkpoints)]
    dx = car_x - current_checkpoint[0]
    dy = car_y - current_checkpoint[1]
    return math.sqrt(dx * dx + dy * dy)

def draw_car(screen):
    if not car_alive:
        return
        
    corners = get_car_corners()
    pygame.draw.polygon(screen, RED, corners)
    
    # Richtung anzeigen
    front_x = car_x + math.cos(math.radians(car_angle)) * car_width/2
    front_y = car_y + math.sin(math.radians(car_angle)) * car_width/2
    pygame.draw.circle(screen, YELLOW, (int(front_x), int(front_y)), 3)

# Strecken-Funktionen
def draw_track(screen, track_points):
    global outer_points, inner_points
    track_width = 80
    # Äußere Begrenzung
    outer_points = []
    inner_points = []
    
    for i, point in enumerate(track_points):
        next_point = track_points[(i + 1) % len(track_points)]
        
        # Normale zur Strecke berechnen
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
    
    # Strecke zeichnen
    pygame.draw.polygon(screen, GRAY, outer_points)
    pygame.draw.polygon(screen, DARK_GRAY, inner_points)
    
    # Mittellinie
    pygame.draw.lines(screen, WHITE, True, track_points, 2)

def draw_checkpoints(screen, checkpoints):
    for checkpoint in checkpoints:
        pygame.draw.circle(screen, GREEN, (int(checkpoint[0]), int(checkpoint[1])), 5)

def point_on_track(point, track_points):
    x, y = point
    track_width = 80
    
    min_distance = float('inf')
    for track_point in track_points:
        dx = x - track_point[0]
        dy = y - track_point[1]
        distance = math.sqrt(dx*dx + dy*dy)
        min_distance = min(min_distance, distance)
    
    return min_distance <= track_width/2

def check_collision(track_points,current_action,state):
    global car_alive, car_reward, reset_timer

    
    car_corners = get_car_corners()
    
    for corner in car_corners:
        if not point_on_track(corner, track_points):
            car_alive = False
            car_reward -= 50  # Strafe für Kollision
            reset_timer = reset_delay  # Timer für automatisches Reset starten
            next_state = getStates()
            train_q_state_learning(state, current_action, car_reward, next_state)
            return True
    return False

def check_checkpoint(checkpoints):
    global car_checkpoints_passed, car_reward
    
    if len(checkpoints) == 0:
        return False
    
    current_checkpoint = checkpoints[car_checkpoints_passed % len(checkpoints)]
    dx = car_x - current_checkpoint[0]
    dy = car_y - current_checkpoint[1]
    distance = math.sqrt(dx*dx + dy*dy)
    
    if distance < 30:  # Checkpoint-Radius
        car_checkpoints_passed += 1
        car_reward += 100  # Belohnung für Checkpoint
        return True
    return False

def reset_game():
    global car_x, car_y, car_angle, car_speed, car_alive, car_distance_traveled
    global car_last_x, car_last_y, car_checkpoints_passed, car_reward, reset_timer
    
    car_x = SCREEN_WIDTH // 2
    car_y = SCREEN_HEIGHT // 2 + 200
    car_angle = 0
    car_speed = 0
    car_alive = True
    car_distance_traveled = 0
    car_last_x = car_x
    car_last_y = car_y
    car_checkpoints_passed = 0
    car_reward = 0
    reset_timer = 0

def draw_ui(screen, font, current_action):
    global reset_timer
    
    # Geschwindigkeit
    speed_text = font.render(f"Geschwindigkeit: {car_speed:.1f}", True, WHITE)
    screen.blit(speed_text, (10, 10))
    
    # Checkpoints
    checkpoint_text = font.render(f"Checkpoints: {car_checkpoints_passed}", True, WHITE)
    screen.blit(checkpoint_text, (10, 50))
    
    # Reward
    reward_text = font.render(f"Reward: {car_reward}", True, WHITE)
    screen.blit(reward_text, (10, 90))
    
    # Status
    status = "Lebendig" if car_alive else "Kollision!"
    status_color = GREEN if car_alive else RED
    status_text = font.render(f"Status: {status}", True, status_color)
    screen.blit(status_text, (10, 130))
    
    # Auto-Reset Timer anzeigen
    if reset_timer > 0:
        countdown = (reset_timer / 60.0)  # Sekunden
        reset_text = font.render(f"Neustart in: {countdown:.1f}s", True, YELLOW)
        screen.blit(reset_text, (10, 170))
    
    # Steuerung
    control_text = font.render("Steuerung: WASD oder Pfeiltasten | R = Manueller Reset", True, WHITE)
    screen.blit(control_text, (10, SCREEN_HEIGHT - 40))
    
    # Aktuelle Aktion anzeigen
    action_names = ["Beschleunigen", "Bremsen", "Links", "Rechts"]
    active_actions = [action_names[i] for i, active in enumerate(current_action) if active]
    action_text = f"Aktionen: {', '.join(active_actions) if active_actions else 'Keine'}"
    action_display = font.render(action_text, True, YELLOW)
    screen.blit(action_display, (10, 210))

def main():
    global car_reward, reset_timer,epsilon, action_index
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("2D Auto Rennspiel - Auto Reset")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    track_points = generate_track()
    checkpoints = generate_checkpoints(track_points)
    
    current_action = [0, 0, 0, 0] 
    
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
            current_action = [0, 0, 0, 0]
            state = getStates()
            old_distance = distance_to_next_checkpoint(checkpoints)
            if random.random() < epsilon:  
                current_action = [0, 0, 0, 0]
                action_index = random.randint(0, 3)
                current_action[action_index] = 1
            else:
                raw_action = model.predict(state.reshape(1, -1), verbose=0)[0]
                action_index = np.argmax(raw_action)
                current_action[action_index] = 1
            update_car(current_action)
            check_collision(track_points,current_action,state)
            next_state = getStates()
            check_checkpoint(checkpoints)
             
            new_distance = distance_to_next_checkpoint(checkpoints)
            delta_distance = old_distance - new_distance
            car_reward += delta_distance * 2
            train_q_state_learning(state,action_index,car_reward,next_state)
            
        else:
            epsilon = max(min_epsilon, epsilon * decay_rate)

        screen.fill(BLACK)
        draw_track(screen, track_points)
        draw_checkpoints(screen, checkpoints)
        draw_car(screen)
        draw_ui(screen, font, current_action)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()

if __name__ == "__main__":
    main()