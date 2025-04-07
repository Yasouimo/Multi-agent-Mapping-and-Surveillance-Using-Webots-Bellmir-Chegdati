from controller import Robot, Emitter, Receiver
import random
import math
import numpy as np
import json

class MazeExplorer(Robot):
    def __init__(self):
        super().__init__()
        
        # Hardware setup
        self.time_step = 64
        self.max_speed = 6.28
        
        # Motors
        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        # Sensors
        self.sensors = [self.getDevice(f'ps{i}') for i in range(8)]
        for sensor in self.sensors:
            sensor.enable(self.time_step)
        
        # Communication devices (for potential future multi-robot expansion)
        self.emitter = self.getDevice("emitter")
        self.receiver = self.getDevice("receiver")
        self.receiver.enable(self.time_step)
        
        # Movement parameters
        self.forward_speed = 0.6 * self.max_speed
        self.turn_speed = 0.8 * self.max_speed
        self.reverse_speed = -0.3 * self.max_speed
        
        # Navigation parameters
        self.avoidance_threshold = 80
        self.collision_threshold = 150
        self.stuck_threshold = 20  # Steps without moving
        
        # Exploration tracking
        self.position = [0, 0]  # Relative position tracking
        self.orientation = 0
        self.explored_map = {}  # Tracks visited locations
        self.last_positions = []  # For loop detection
        self.steps_without_move = 0
        self.stuck_count = 0
        
        # Action space
        self.actions = {
            "forward": (self.forward_speed, self.forward_speed),
            "left_90": (-self.turn_speed, self.turn_speed),
            "right_90": (self.turn_speed, -self.turn_speed),
            "left_180": (-self.turn_speed, self.turn_speed),
            "right_180": (self.turn_speed, -self.turn_speed),
            "reverse": (self.reverse_speed, self.reverse_speed)
        }
        
        # Action durations (ms)
        self.action_durations = {
            "forward": 1000,
            "left_90": 500,
            "right_90": 500,
            "left_180": 1000,
            "right_180": 1000,
            "reverse": 300
        }
        
        # Initialize movement
        self.current_action = None
        self.action_start_time = 0
    
    def get_sensor_state(self):
        """Process sensor readings into a simplified state"""
        values = [s.getValue() for s in self.sensors]
        return {
            'front': max(values[0], values[7]),
            'left': max(values[5], values[6]),
            'right': max(values[1], values[2]),
            'back': max(values[3], values[4])
        }
    
    def is_blocked(self, sensor_state):
        """Check if path is blocked"""
        return (sensor_state['front'] > self.avoidance_threshold or
                sensor_state['left'] > self.collision_threshold or
                sensor_state['right'] > self.collision_threshold)
    
    def get_position_key(self):
        """Create a discrete position key for mapping"""
        return (round(self.position[0] / 0.1), round(self.position[1] / 0.1))
    
    def update_position(self, left_speed, right_speed, duration):
        """Update estimated position based on movement"""
        # Convert duration to seconds
        t = duration / 1000.0
        
        # Straight movement
        if abs(left_speed - right_speed) < 0.1:
            distance = ((left_speed + right_speed) / 2) * t
            self.position[0] += distance * math.cos(self.orientation)
            self.position[1] += distance * math.sin(self.orientation)
        # Turning
        else:
            rotation = (right_speed - left_speed) * t / 0.02  # 0.02 is approx wheel distance
            self.orientation = (self.orientation + rotation) % (2 * math.pi)
        
        # Track recent positions for loop detection
        pos_key = self.get_position_key()
        self.last_positions.append(pos_key)
        if len(self.last_positions) > 10:
            self.last_positions.pop(0)
        
        # Check if we're moving
        if len(self.last_positions) > 5 and len(set(self.last_positions[-5:])) == 1:
            self.steps_without_move += 1
        else:
            self.steps_without_move = 0
    
    def is_in_loop(self):
        """Detect if robot is going in circles"""
        if len(self.last_positions) < 10:
            return False
        
        # Check if we've visited the same places recently
        unique_positions = len(set(self.last_positions))
        return unique_positions < 5
    
    def should_explore_new_area(self):
        """Determine if we should prioritize exploration"""
        pos_key = self.get_position_key()
        
        # Count how many times we've been here
        visit_count = sum(1 for pos in self.last_positions if pos == pos_key)
        
        # If we've been here too much, try something new
        return visit_count > 3
    
    def choose_action(self, sensor_state):
        """Select an action based on current state"""
        pos_key = self.get_position_key()
        
        # Emergency collision avoidance
        if sensor_state['front'] > self.collision_threshold:
            if sensor_state['left'] < sensor_state['right']:
                return "left_180"
            else:
                return "right_180"
        
        # If stuck, try to get unstuck
        if self.steps_without_move > self.stuck_threshold:
            self.stuck_count += 1
            if self.stuck_count % 2 == 0:
                return "left_180"
            else:
                return "right_180"
        
        # Avoid obstacles
        if sensor_state['front'] > self.avoidance_threshold:
            if sensor_state['left'] < sensor_state['right']:
                return "left_90"
            else:
                return "right_90"
        
        # If we're in a loop or visiting the same area too much
        if self.is_in_loop() or self.should_explore_new_area():
            return random.choice(["left_90", "right_90", "left_180", "right_180"])
        
        # Default to forward motion
        return "forward"
    
    def execute_action(self, action_name):
        """Execute the selected action"""
        left_speed, right_speed = self.actions[action_name]
        duration = self.action_durations[action_name]
        
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
        
        # Update position estimate
        self.update_position(left_speed, right_speed, duration)
        
        # Mark current position as explored
        self.explored_map[self.get_position_key()] = True
        
        # Perform the action for its duration
        start_time = self.getTime()
        while self.getTime() - start_time < duration / 1000.0:
            if self.step(self.time_step) == -1:
                break
        
        # Stop after action completion
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.step(self.time_step)
    
    def run(self):
        while self.step(self.time_step) != -1:
            # Read sensors and get state
            sensor_state = self.get_sensor_state()
            
            # Choose and execute action
            action = self.choose_action(sensor_state)
            self.execute_action(action)
            
            # Reset stuck counter if we moved
            if action in ["forward", "left_90", "right_90"]:
                if self.steps_without_move == 0:
                    self.stuck_count = 0
            
            # Print status occasionally
            if random.random() < 0.05:
                pos = self.get_position_key()
                explored = len(self.explored_map)
                print(f"At {pos} | Action: {action} | Explored: {explored} areas | Steps stuck: {self.steps_without_move}")

# Create and run the robot
robot = MazeExplorer()
robot.run()