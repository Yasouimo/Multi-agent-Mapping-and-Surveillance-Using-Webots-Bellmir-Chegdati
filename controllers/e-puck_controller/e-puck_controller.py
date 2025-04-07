from controller import Robot
import random
import math

# Initialize the robot
robot = Robot()

# Time step for the simulation
time_step = int(robot.getBasicTimeStep())

# Max motor speed (e-puck max speed is ~6.28 rad/s)
max_speed = 6.28

# Get motors
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

# Enable motors for velocity control
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Get distance sensors (8 sensors on e-puck)
num_sensors = 8
sensors = [robot.getDevice(f'ps{i}') for i in range(num_sensors)]
for sensor in sensors:
    sensor.enable(time_step)

# ULTRA HIGH SPEED SETTINGS - MUCH FASTER THAN ORIGINAL
forward_speed = 0.9 * max_speed  # Nearly full speed forward
turn_speed = 0.8 * max_speed     # Fast turning
reverse_speed = -0.7 * max_speed # Fast reverse

# Quick detection thresholds
OBSTACLE_THRESHOLD = 70  # Detect obstacles
WALL_THRESHOLD = 130     # Detect walls

# Simple path memory (just grid coordinates)
visited_cells = {}
robot_x, robot_y = 0, 0
cell_size = 0.2  # 20cm grid cells

# Simplified Q-learning
Q_table = {}
alpha = 0.5      # Fast learning rate
gamma = 0.8      # Discount factor
epsilon = 0.3    # Exploration rate
min_epsilon = 0.1

# Actions: Forward, Left, Right
actions = ["Forward", "Left", "Right"]

# Get basic state from sensors
def get_state(sensor_values):
    # Only use essential sensors: front, left, right
    front = max(sensor_values[0], sensor_values[7])
    left = max(sensor_values[5], sensor_values[6])
    right = max(sensor_values[1], sensor_values[2])
    
    # Simple binary state
    state = []
    for value in [front, left, right]:
        if value < OBSTACLE_THRESHOLD:
            state.append(0)  # Clear
        else:
            state.append(1)  # Obstacle
    
    return tuple(state)

# Main loop
step_count = 0
current_action = "Forward"  # Default starting action

while robot.step(time_step) != -1:
    step_count += 1
    
    # Read sensors
    sensor_values = [sensor.getValue() for sensor in sensors]
    
    # Get current state
    state = get_state(sensor_values)
    
    # Update grid position (very simple odometry)
    # This just approximates position for path tracking purposes
    current_cell = (int(robot_x/cell_size), int(robot_y/cell_size))
    visited_cells[current_cell] = visited_cells.get(current_cell, 0) + 1
    
    # Check if we're about to hit something
    front_obstacle = max(sensor_values[0], sensor_values[7]) > WALL_THRESHOLD
    left_obstacle = max(sensor_values[5], sensor_values[6]) > OBSTACLE_THRESHOLD
    right_obstacle = max(sensor_values[1], sensor_values[2]) > OBSTACLE_THRESHOLD
    
    # Reset the action for this iteration
    action_taken = None
    
    # SIMPLIFIED DIRECT CONTROL LOGIC
    if front_obstacle:  # About to hit wall - emergency maneuver
        # Back up quickly
        left_motor.setVelocity(reverse_speed)
        right_motor.setVelocity(reverse_speed)
        robot.step(time_step * 3)  # Just a few steps back
        
        # Turn away from obstacle (random direction if unsure)
        if left_obstacle and not right_obstacle:
            # Turn right if left is blocked
            left_motor.setVelocity(turn_speed)
            right_motor.setVelocity(-turn_speed * 0.5)
            action_taken = "Right"
        elif right_obstacle and not left_obstacle:
            # Turn left if right is blocked
            left_motor.setVelocity(-turn_speed * 0.5)
            right_motor.setVelocity(turn_speed)
            action_taken = "Left"
        else:
            # Random sharp turn
            if random.random() < 0.5:
                left_motor.setVelocity(turn_speed)
                right_motor.setVelocity(-turn_speed * 0.5)
                action_taken = "Right"
            else:
                left_motor.setVelocity(-turn_speed * 0.5)
                right_motor.setVelocity(turn_speed)
                action_taken = "Left"
        
        # Turn for a shorter time
        robot.step(time_step * 5)
        
        # Update position estimate after maneuver
        heading_change = random.uniform(-math.pi/4, math.pi/4)  # Approximate heading change
        robot_x += random.uniform(-0.05, 0.05)  # Small position adjustment
        robot_y += random.uniform(-0.05, 0.05)
    
    elif left_obstacle and not right_obstacle:
        # Turn slightly right while moving forward
        left_motor.setVelocity(forward_speed)
        right_motor.setVelocity(forward_speed * 0.4)
        action_taken = "Right"
        
        # Update position estimate
        robot_x += 0.01 * math.cos(math.pi/8)
        robot_y += 0.01 * math.sin(math.pi/8)
    
    elif right_obstacle and not left_obstacle:
        # Turn slightly left while moving forward
        left_motor.setVelocity(forward_speed * 0.4)
        right_motor.setVelocity(forward_speed)
        action_taken = "Left"
        
        # Update position estimate
        robot_x += 0.01 * math.cos(-math.pi/8)
        robot_y += 0.01 * math.sin(-math.pi/8)
    
    else:
        # Use Q-learning for general exploration when no immediate obstacles
        if random.random() < epsilon:  # Explore
            # Prefer exploring new areas
            if visited_cells.get(current_cell, 0) > 5:  # Well-visited area
                # More likely to turn in familiar areas
                action_taken = random.choice(["Left", "Right", "Forward", "Forward"])
            else:
                # More likely to go straight in new areas
                action_taken = random.choice(["Forward", "Forward", "Forward", "Left", "Right"])
        else:  # Exploit
            if state not in Q_table:
                Q_table[state] = {"Forward": 1.0, "Left": 0.5, "Right": 0.5}
            
            action_taken = max(Q_table[state].items(), key=lambda x: x[1])[0]
        
        # Execute selected action at high speed
        if action_taken == "Forward":
            left_motor.setVelocity(forward_speed)
            right_motor.setVelocity(forward_speed)
            robot_x += 0.02 * math.cos(0)
            robot_y += 0.02 * math.sin(0)
        elif action_taken == "Left":
            left_motor.setVelocity(forward_speed * 0.2)
            right_motor.setVelocity(forward_speed)
            robot_x += 0.01 * math.cos(-math.pi/6)
            robot_y += 0.01 * math.sin(-math.pi/6)
        elif action_taken == "Right":
            left_motor.setVelocity(forward_speed)
            right_motor.setVelocity(forward_speed * 0.2)
            robot_x += 0.01 * math.cos(math.pi/6)
            robot_y += 0.01 * math.sin(math.pi/6)
    
    # Store the current action for Q-learning
    current_action = action_taken if action_taken is not None else current_action
    
    # Very simple Q-learning update (only when needed)
    if step_count % 5 == 0 and not front_obstacle:
        # Get new sensor readings
        new_sensor_values = [sensor.getValue() for sensor in sensors]
        new_state = get_state(new_sensor_values)
        
        # Initialize Q values if needed
        if state not in Q_table:
            Q_table[state] = {"Forward": 1.0, "Left": 0.5, "Right": 0.5}
        if new_state not in Q_table:
            Q_table[new_state] = {"Forward": 1.0, "Left": 0.5, "Right": 0.5}
        
        # Calculate reward
        if front_obstacle:
            reward = -5
        elif left_obstacle or right_obstacle:
            reward = -1
        else:
            # Reward for exploring new areas
            if visited_cells.get(current_cell, 0) <= 2:
                reward = 3
            else:
                reward = 1
        
        # Update Q-values using current_action
        best_next_action = max(Q_table[new_state].items(), key=lambda x: x[1])[0]
        Q_table[state][current_action] += alpha * (reward + gamma * Q_table[new_state][best_next_action] - Q_table[state][current_action])
    
    # Decay exploration rate
    if step_count % 100 == 0:
        epsilon = max(min_epsilon, epsilon * 0.99)
    
    # Debug output
    if step_count % 500 == 0:
        unique_cells = len(visited_cells)
        print(f"Step: {step_count}, Explored: {unique_cells} cells")