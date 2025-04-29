from controller import Robot
import random
import math
import os
import sys
from robot_communication import RobotCommunicator

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

# Initialize robot communicator
communicator = RobotCommunicator(robot)
print("Robot communication initialized")

# Get distance sensors (8 sensors on e-puck)
num_sensors = 8
sensors = [robot.getDevice(f'ps{i}') for i in range(num_sensors)]
for sensor in sensors:
    sensor.enable(time_step)

# ULTRA HIGH SPEED SETTINGS - EXACTLY AS ORIGINAL
forward_speed = 0.9 * max_speed  # Nearly full speed forward
turn_speed = 0.8 * max_speed     # Fast turning
reverse_speed = -0.7 * max_speed # Fast reverse

# Quick detection thresholds
OBSTACLE_THRESHOLD = 70  # Detect obstacles
WALL_THRESHOLD = 130     # Detect walls

# Simple path memory (just grid coordinates)
visited_cells = {}
robot_x, robot_y = 0, 0
heading = 0.0  # Track robot heading in radians
cell_size = 0.2  # 20cm grid cells

# Simplified Q-learning
Q_table = {}
alpha = 0.5      # Fast learning rate
gamma = 0.8      # Discount factor
epsilon = 0.3    # Exploration rate
min_epsilon = 0.1

# Actions: Forward, Left, Right
actions = ["Forward", "Left", "Right"]

# Setup detection system
try:
    from detection import ObjectDetector
    # Pass the communicator instance to the detector so it can use our communication system
    detector = ObjectDetector(robot, communicator=communicator)
    detection_enabled = True
    print("Object detection initialized with communicator integration")
except Exception as e:
    print(f"Could not initialize detector: {e}")
    detection_enabled = False

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

# Handle messages from other robots
def handle_robot_messages(messages):
    global robot_x, robot_y, epsilon, Q_table
    
    if not messages:
        return
        
    for message in messages:
        if message["type"] == "detection":
            detection = message["detection"]
            robot_id = message["robot_id"]
            
            # Process all detections whether alarm is activated or not
            print(f"Robot {robot_id} detected a {detection['object_type']} with confidence {detection['confidence']:.2f}")
            
            # If this is a cat detection with high confidence
            if detection["object_type"] == "Cat" and detection["confidence"] > 0.5:
                if not communicator.alarm_activated:
                    print(f"🚨 Alarm activated by Robot {robot_id}'s cat detection!")
                    communicator.alarm_activated = True
                    communicator.first_detector = robot_id
                    robot_status = "detected_object"
                
                if "location" in detection:
                    # Get direction to the cat
                    cat_x = detection["location"]["x"]
                    cat_y = detection["location"]["y"]
                    
                    # Make this area more attractive in our Q-table
                    # For simplicity, we just decrease epsilon to be more exploitative
                    print(f"🐱 Cat detected by another robot! Location: ({cat_x:.2f}, {cat_y:.2f})")
                    epsilon = max(min_epsilon, epsilon * 0.8)  # Become more exploitative
                    
                    # Calculate direction from robot to cat
                    dx = cat_x - robot_x
                    dy = cat_y - robot_y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance < 3.0:  # Only if within reasonable range
                        # Create a virtual "reward" in Q-table for heading toward cat location
                        target_heading = math.atan2(dy, dx)
                        heading_diff = abs(heading - target_heading)
                        
                        # Normalize heading difference
                        if heading_diff > math.pi:
                            heading_diff = 2 * math.pi - heading_diff
                            
                        # Adjust Q-table to favor moving toward cat
                        for state in Q_table.keys():
                            if heading_diff < math.pi/4:  # Cat is roughly ahead
                                Q_table[state]["Forward"] *= 1.2  # Boost going forward
                            elif heading_diff < math.pi/2:  # Cat is to the right
                                Q_table[state]["Right"] *= 1.2  # Boost turning right
                            else:  # Cat is to the left
                                Q_table[state]["Left"] *= 1.2  # Boost turning left
                                
        elif message["type"] == "command":
            # Check for first detection commands
            if message["command"]["action"] == "first_detection":
                communicator.alarm_activated = True
                first_detector = message["command"]["params"]["robot_name"]
                print(f"Acknowledging first detection by Robot {first_detector}")
            
            # Another robot is sending a command
            command = message["command"]
            robot_id = message["robot_id"]
            
            if command["action"] == "investigate":
                print(f"Received investigation command from robot {robot_id}")
                
                # Extract target location if provided
                if "params" in command and "location" in command["params"]:
                    target_x = command["params"]["location"]["x"]
                    target_y = command["params"]["location"]["y"]
                    
                    print(f"Investigation target: ({target_x:.2f}, {target_y:.2f})")
                    
                    # Adjust exploration based on priority
                    if "priority" in command["params"] and command["params"]["priority"] == "high":
                        epsilon = max(min_epsilon, epsilon * 0.7)  # Become significantly more exploitative
                        print("High priority investigation - reducing exploration")
            
            elif command["action"] == "avoid":
                print(f"Received avoidance command from robot {robot_id}")
                
                # Increase exploration to encourage finding alternative paths
                epsilon = min(0.5, epsilon * 1.2)
                print("Increasing exploration rate to find alternative routes")
                
        elif message["type"] == "position":
            # Another robot's position update - we can use this to avoid collisions
            # or coordinate exploration efforts
            robot_id = message["robot_id"]
            other_x = message["position"]["x"]
            other_y = message["position"]["y"]
            other_status = message["status"]
            
            # Simple collision avoidance if robots are too close
            dx = other_x - robot_x
            dy = other_y - robot_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # If too close to another robot, adjust Q-table to encourage separation
            if distance < 0.4:  # Too close, less than 40cm
                # Determine direction to other robot
                direction = math.atan2(dy, dx)
                
                # Adjust Q-table to avoid moving toward other robot
                for state in Q_table.keys():
                    # If other robot is ahead, discourage going forward
                    if abs(heading - direction) < math.pi/4:
                        Q_table[state]["Forward"] *= 0.8  # Reduce probability of moving forward
                        print(f"Too close to robot {robot_id} - avoiding collision")
    
    return

# Check if we should react to specific detected objects
def react_to_detections():
    global epsilon, robot_status
    
    # Check for recent cat detections in our communication system
    recent_cats = communicator.get_recent_detections(detection_type="Cat", max_age=20)
    
    if recent_cats:
        # We have recent cat detections, prioritize investigation
        most_recent = sorted(recent_cats, key=lambda x: x["timestamp"], reverse=True)[0]
        robot_id = most_recent["robot_id"]
        
        print(f"Reacting to recent Cat detection from robot {robot_id}")
        
        # Check if location information is available
        if "location" in most_recent:
            cat_x = most_recent["location"]["x"]
            cat_y = most_recent["location"]["y"]
            
            # Calculate direction from robot to cat
            dx = cat_x - robot_x
            dy = cat_y - robot_y
            target_heading = math.atan2(dy, dx)
            
            # Simple heading adjustment to move toward cat
            heading_diff = heading - target_heading
            
            # Normalize heading difference
            if heading_diff > math.pi:
                heading_diff = heading_diff - 2*math.pi
            elif heading_diff < -math.pi:
                heading_diff = heading_diff + 2*math.pi
                
            # Set appropriate motor speeds to move toward cat
            if abs(heading_diff) < math.pi/6:
                # Cat is roughly ahead, move forward
                left_motor.setVelocity(forward_speed)
                right_motor.setVelocity(forward_speed)
                robot_status = "investigating"
                return True
            elif heading_diff > 0:
                # Cat is to the left, turn left
                left_motor.setVelocity(forward_speed * 0.2)
                right_motor.setVelocity(forward_speed)
                robot_status = "investigating"
                return True
            else:
                # Cat is to the right, turn right
                left_motor.setVelocity(forward_speed)
                right_motor.setVelocity(forward_speed * 0.2)
                robot_status = "investigating"
                return True
    
    return False

# Main loop
print("Starting robot controller with enhanced communication and detection integration")
step_count = 0
current_action = "Forward"  # Default starting action
robot_status = "exploring"
broadcast_interval = 2.0  # How often to broadcast position
last_broadcast_time = 0

while robot.step(time_step) != -1:
    step_count += 1
    
    # Run object detection (very infrequently to maintain speed)
    if detection_enabled and step_count % 100 == 0:
        # Pass robot position to the detector and update
        alarm_triggered = detector.update(step_count, robot_x, robot_y)
        
        if alarm_triggered:  # Remove the not communicator.alarm_activated check
            print(f"🚨 Alarm triggered by Robot {robot.getName()}!")
            robot_status = "detected_object"
            if not communicator.alarm_activated:  # Only set first detector if not already set
                communicator.alarm_activated = True
                communicator.first_detector = robot.getName()
                # Broadcast that we were first to detect
                communicator.broadcast_command("first_detection", {
                    "robot_name": robot.getName(),
                    "timestamp": step_count
                })
    
    # Update robot communication and handle incoming messages
    # The update function now handles regular position broadcasting based on interval
    new_messages = communicator.update(robot_x, robot_y, heading, robot_status)
    if new_messages:
        print(f"Received {len(new_messages)} new messages from other robots")
        handle_robot_messages(new_messages)
    
    # Read sensors
    sensor_values = [sensor.getValue() for sensor in sensors]
    
    # Get current state
    state = get_state(sensor_values)
    
    # Update grid position (very simple odometry)
    current_cell = (int(robot_x/cell_size), int(robot_y/cell_size))
    visited_cells[current_cell] = visited_cells.get(current_cell, 0) + 1
    
    # Check if we should react to recent detections from other robots
    # This takes precedence over normal movement behavior
    if react_to_detections():
        continue
    
    # Check if we're about to hit something
    front_obstacle = max(sensor_values[0], sensor_values[7]) > WALL_THRESHOLD
    left_obstacle = max(sensor_values[5], sensor_values[6]) > OBSTACLE_THRESHOLD
    right_obstacle = max(sensor_values[1], sensor_values[2]) > OBSTACLE_THRESHOLD
    
    # Reset the action for this iteration
    action_taken = None
    
    # Check if we should respond to a nearby robot's detection
    nearby_robots = communicator.get_nearby_robots()
    for robot_id, info in nearby_robots.items():
        if info["status"] == "detected_object":
            # If another robot detected something, we might want to help
            if random.random() < 0.3:  # 30% chance to go help
                print(f"Moving to assist robot {robot_id} with detection")
                
                # Get position of the other robot
                other_pos = info["position"]
                other_x, other_y = other_pos["x"], other_pos["y"]
                
                # Calculate direction from our robot to the other robot
                dx = other_x - robot_x
                dy = other_y - robot_y
                target_heading = math.atan2(dy, dx)
                
                # Simple heading adjustment to move toward other robot
                heading_diff = heading - target_heading
                
                # Normalize heading difference
                if heading_diff > math.pi:
                    heading_diff = heading_diff - 2*math.pi
                elif heading_diff < -math.pi:
                    heading_diff = heading_diff + 2*math.pi
                    
                # Set appropriate motor speeds to move toward other robot
                if abs(heading_diff) < math.pi/6:
                    # Other robot is roughly ahead, move forward
                    left_motor.setVelocity(forward_speed)
                    right_motor.setVelocity(forward_speed)
                    robot_status = "assisting"
                    continue
                elif heading_diff > 0:
                    # Other robot is to the left, turn left
                    left_motor.setVelocity(forward_speed * 0.2)
                    right_motor.setVelocity(forward_speed)
                    robot_status = "assisting"
                    continue
                else:
                    # Other robot is to the right, turn right
                    left_motor.setVelocity(forward_speed)
                    right_motor.setVelocity(forward_speed * 0.2)
                    robot_status = "assisting"
                    continue
    
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
            heading += math.pi/4  # Approximate heading change
        elif right_obstacle and not left_obstacle:
            # Turn left if right is blocked
            left_motor.setVelocity(-turn_speed * 0.5)
            right_motor.setVelocity(turn_speed)
            action_taken = "Left"
            heading -= math.pi/4  # Approximate heading change
        else:
            # Random sharp turn
            if random.random() < 0.5:
                left_motor.setVelocity(turn_speed)
                right_motor.setVelocity(-turn_speed * 0.5)
                action_taken = "Right"
                heading += math.pi/4  # Approximate heading change
            else:
                left_motor.setVelocity(-turn_speed * 0.5)
                right_motor.setVelocity(turn_speed)
                action_taken = "Left"
                heading -= math.pi/4  # Approximate heading change
        
        # Turn for a shorter time
        robot.step(time_step * 5)
        
        # Update position estimate after maneuver
        heading_change = random.uniform(-math.pi/4, math.pi/4)  # Approximate heading change
        heading += heading_change
        robot_x += random.uniform(-0.05, 0.05)  # Small position adjustment
        robot_y += random.uniform(-0.05, 0.05)
        
        # Broadcast an obstacle detection to warn other robots
        obstacle_loc = {
            "x": robot_x + 0.3 * math.cos(heading),
            "y": robot_y + 0.3 * math.sin(heading)
        }
        communicator.broadcast_detection("Obstacle", 0.9, obstacle_loc)
        
        # Send a command to other robots to avoid this area
        communicator.broadcast_command("avoid", {
            "location": obstacle_loc,
            "radius": 0.5,
            "reason": "obstacle"
        })
        
        # Normalize heading to keep it within -π to π
        heading = math.atan2(math.sin(heading), math.cos(heading))
    
    elif left_obstacle and not right_obstacle:
        # Turn slightly right while moving forward
        left_motor.setVelocity(forward_speed)
        right_motor.setVelocity(forward_speed * 0.4)
        action_taken = "Right"
        
        # Update position estimate
        heading += math.pi/16  # Small heading adjustment
        robot_x += 0.01 * math.cos(heading)
        robot_y += 0.01 * math.sin(heading)
    
    elif right_obstacle and not left_obstacle:
        # Turn slightly left while moving forward
        left_motor.setVelocity(forward_speed * 0.4)
        right_motor.setVelocity(forward_speed)
        action_taken = "Left"
        
        # Update position estimate
        heading -= math.pi/16  # Small heading adjustment
        robot_x += 0.01 * math.cos(heading)
        robot_y += 0.01 * math.sin(heading)
    
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
            robot_x += 0.02 * math.cos(heading)
            robot_y += 0.02 * math.sin(heading)
        elif action_taken == "Left":
            left_motor.setVelocity(forward_speed * 0.2)
            right_motor.setVelocity(forward_speed)
            heading -= math.pi/12
            robot_x += 0.01 * math.cos(heading)
            robot_y += 0.01 * math.sin(heading)
        elif action_taken == "Right":
            left_motor.setVelocity(forward_speed)
            right_motor.setVelocity(forward_speed * 0.2)
            heading += math.pi/12
            robot_x += 0.01 * math.cos(heading)
            robot_y += 0.01 * math.sin(heading)
    
    # Normalize heading to keep it within -π to π
    heading = math.atan2(math.sin(heading), math.cos(heading))
    
    # Store the current action for Q-learning
    current_action = action_taken if action_taken is not None else current_action
    
    # Reset status if we were in a special state
    if robot_status != "exploring" and step_count % 50 == 0:
        robot_status = "exploring"
    
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
        nearby_count = len(communicator.get_nearby_robots())
        print(f"Step: {step_count}, Explored: {unique_cells} cells, Nearby robots: {nearby_count}")
        
        # Get collaborative detection information
        recent_detections = communicator.get_recent_detections(max_age=30)
        if recent_detections:
            print(f"Recent network detections: {len(recent_detections)}")
            for detection in recent_detections[:3]:  # Show at most 3
                robot_id = detection["robot_id"]
                obj_type = detection["type"]
                conf = detection["confidence"]
                print(f"  - Robot {robot_id}: {obj_type} ({conf:.2f})")