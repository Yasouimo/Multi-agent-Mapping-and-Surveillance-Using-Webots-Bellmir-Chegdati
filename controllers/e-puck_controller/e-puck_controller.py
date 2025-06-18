# e-puck_controller.py

from controller import Robot
import numpy as np
import math
import glob
import pickle
import os
import time
from robot_communication import RobotCommunicator
from detection import ObjectDetector
from cooperative_mapping import CooperativeMapping
# <<< NOTE >>>: The visualizer is removed to focus on the simple, direct proof.
# from cooperation_visualizer import CooperationVisualizer

class EPuckController:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        self.max_speed = 6.28
        self.robot_name = self.robot.getName()

        # --- Motor and Sensor Setup ---
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        self.left_encoder = self.robot.getDevice("left wheel sensor")
        self.right_encoder = self.robot.getDevice("right wheel sensor")
        self.left_encoder.enable(self.time_step)
        self.right_encoder.enable(self.time_step)

        self.sensors = [self.robot.getDevice(f'ps{i}') for i in range(8)]
        for sensor in self.sensors:
            sensor.enable(self.time_step)

        # --- Core Modules ---
        self.communicator = RobotCommunicator(self.robot)
        self.detector = ObjectDetector(self.robot, self.communicator)
        # <<< MODIFIED >>>: Pass the robot's name to the mapping module
        self.mapping = CooperativeMapping(self.robot_name)

        # <<< ADDED >>>: A dedicated target for cooperative actions.
        # This will override any self-generated target.
        self.cooperative_target = None
        self.target_threshold = 0.15 # Slightly larger threshold for cooperative target

        # --- Odometry ---
        self.position = np.array([0.0, 0.0])
        self.orientation = 0.0
        self.prev_left_encoder = 0.0
        self.prev_right_encoder = 0.0
        self.wheel_radius = 0.02
        self.axle_length = 0.052

        # --- State Machine ---
        self.state = "EXPLORING" # Simplified state machine
        self.avoidance_step = 0
        
        print(f"[{self.robot_name}] Simple Cooperative Controller initialized.")

    def set_motor_speeds(self, left, right):
        self.left_motor.setVelocity(np.clip(left, -self.max_speed, self.max_speed))
        self.right_motor.setVelocity(np.clip(right, -self.max_speed, self.max_speed))

    def update_pose(self):
        left_val, right_val = self.left_encoder.getValue(), self.right_encoder.getValue()
        delta_left = (left_val - self.prev_left_encoder) * self.wheel_radius
        delta_right = (right_val - self.prev_right_encoder) * self.wheel_radius
        self.prev_left_encoder, self.prev_right_encoder = left_val, right_val
        distance = (delta_left + delta_right) / 2.0
        delta_orientation = (delta_right - delta_left) / self.axle_length
        self.orientation = (self.orientation + delta_orientation) % (2 * np.pi)
        self.position[0] += distance * math.cos(self.orientation)
        self.position[1] += distance * math.sin(self.orientation)

    # <<< ADDED >>>: Function to check for and handle cooperative messages
    def check_cooperative_messages(self):
        messages = self.communicator.check_for_messages()
        for msg in messages:
            # Check for a detection message from another robot
            if msg.get("type") == "detection" and self.robot_name != msg.get("robot_name"):
                # To prevent all robots rushing to the same spot, let's have only one respond.
                # A simple rule: The next robot in line by name will respond.
                # This is a simple but effective coordination rule.
                
                # A more robust way to get robot index
                try:
                    my_id = int(self.robot_name.replace("e-puck", "").replace("(", "").replace(")", "")) if "e-puck(" in self.robot_name else 0
                    sender_id = int(msg["robot_name"].replace("e-puck", "").replace("(", "").replace(")", "")) if "e-puck(" in msg["robot_name"] else 0

                    # Let's assume 4 robots for this logic, adjust if you have a different number
                    num_robots = 4 
                    if my_id == (sender_id + 1) % num_robots:
                        pos = msg.get("position")
                        if pos:
                            self.cooperative_target = np.array(pos)
                            print("="*60)
                            print(f"[{self.robot_name}] COOPERATION: Received detection from [{msg['robot_name']}]!")
                            print(f"[{self.robot_name}] Abandoning my task to investigate location {pos}.")
                            print("="*60)
                except ValueError:
                    # Fallback for names that don't fit the pattern
                    pass


    def run(self):
        """Main control loop with simplified cooperative logic."""
        step_count = 0
        # <<< MODIFIED >>>: Use the detection interval from the detector module for sync timing
        sync_interval = self.detector.detection_interval 
        
        while self.robot.step(self.time_step) != -1:
            step_count += 1
            self.update_pose()
            
            # --- Always check for cooperative messages ---
            self.check_cooperative_messages()

            # --- High-Priority Cooperative Action ---
            # <<< MODIFIED >>>: This block is the core of the new cooperative behavior.
            # If we have a cooperative target, we handle it above all else.
            if self.cooperative_target is not None:
                target_vector = self.cooperative_target - self.position
                distance_to_target = np.linalg.norm(target_vector)

                # If we've arrived, clear the cooperative target and go back to normal work.
                if distance_to_target < self.target_threshold:
                    print(f"[{self.robot_name}] Arrived at cooperative target. Resuming exploration.")
                    self.cooperative_target = None
                    self.state = "EXPLORING"
                    self.set_motor_speeds(0, 0)
                else:
                    # Steer towards the cooperative target
                    target_angle = math.atan2(target_vector[1], target_vector[0])
                    angle_diff = target_angle - self.orientation
                    # Normalize angle
                    while angle_diff > np.pi: angle_diff -= 2 * np.pi
                    while angle_diff < -np.pi: angle_diff += 2 * np.pi
                    
                    steer = np.clip(angle_diff, -2.0, 2.0)
                    self.set_motor_speeds(self.max_speed * 0.8 - steer, self.max_speed * 0.8 + steer)
                
                # Skip the normal state machine while handling a cooperative task
                continue

            # --- Normal State Machine (Simplified) ---
            sensor_values = [s.getValue() for s in self.sensors]

            # Obstacle Avoidance takes precedence over exploration
            is_obstacle = sensor_values[0] > 150 or sensor_values[7] > 150
            if is_obstacle or self.state == "AVOIDING":
                self.state = "AVOIDING"
                # Simple avoidance: back up and turn
                if self.avoidance_step < 15:
                    self.set_motor_speeds(-self.max_speed * 0.5, -self.max_speed * 0.5)
                elif self.avoidance_step < 45:
                    self.set_motor_speeds(self.max_speed * 0.5, -self.max_speed * 0.5)
                else:
                    self.state = "EXPLORING" # Finished avoiding
                    self.avoidance_step = 0
                self.avoidance_step += 1
            else:
                self.state = "EXPLORING"
                # Simple exploration: drive forward with slight correction from side sensors
                side_steer = (sensor_values[5] - sensor_values[2]) / 500.0
                self.set_motor_speeds(self.max_speed - side_steer, self.max_speed + side_steer)

            # --- Sensor and Detection Updates ---
            # These run in the background regardless of state
            self.detector.update(step_count, self.position[0], self.position[1])
            self.mapping.update_map_from_sensors(self.position, self.orientation, sensor_values)
            
            # <<< CRITICAL FIX & ADDITION >>>
            # This block was missing. It periodically saves this robot's data and loads
            # data from other robots, which is the core of cooperative mapping.
            if step_count > 0 and step_count % sync_interval == 0:
                print(f"[{self.robot_name}] Syncing data with team...")
                self.mapping.sync_data(
                    my_pos=self.position,
                    my_target=self.cooperative_target,
                    my_robot_name=self.robot_name,
                    current_state=self.state,
                    new_detections=self.detector.last_detections
                )
                # Clear the detections after they have been synced
                self.detector.last_detections.clear()


if __name__ == "__main__":
    controller = EPuckController()
    controller.run()