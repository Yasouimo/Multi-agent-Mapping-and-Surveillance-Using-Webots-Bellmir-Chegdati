from controller import Robot
import numpy as np
import math
from robot_communication import RobotCommunicator
from detection import ObjectDetector
from cooperative_mapping import CooperativeMapping

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
        self.mapping = CooperativeMapping(self.robot_name)

        # --- Odometry ---
        self.position = np.array([0.0, 0.0])
        self.orientation = 0.0
        self.prev_left_encoder = 0.0
        self.prev_right_encoder = 0.0
        self.wheel_radius = 0.02
        self.axle_length = 0.052

        # --- Definitive 4-State Machine ---
        self.state = "INITIAL_MANEUVER"
        self.initial_maneuver_steps = 75
        self.avoidance_step = 0
        self.current_target = None
        self.target_threshold = 0.1

        print(f"[{self.robot_name}] Definitive 4-State controller initialized.")

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

    def run(self):
        """Main control loop with the 4-state machine."""
        step_count = 0
        while self.robot.step(self.time_step) != -1:
            step_count += 1
            self.update_pose()
            self.mapping.update_map_from_sensors(self.position, self.orientation, [s.getValue() for s in self.sensors])

            if step_count % 50 == 0:
                self.mapping.sync_data(self.position, self.current_target, self.robot_name)

            # --- STATE MACHINE LOGIC ---
            if self.state == "INITIAL_MANEUVER":
                if step_count < self.initial_maneuver_steps:
                    self.set_motor_speeds(self.max_speed, self.max_speed)
                else:
                    print(f"[{self.robot_name}] Initial maneuver complete.")
                    self.set_motor_speeds(0, 0)
                    self.state = "SELECTING_GOAL"

            elif self.state == "SELECTING_GOAL":
                self.current_target = self.mapping.assign_exploration_target(self.position, self.robot_name)
                if self.current_target:
                    print(f"[{self.robot_name}] New target acquired. Following.")
                    self.state = "FOLLOWING_TARGET"
                else: # No valid target found, spin to scan
                    self.set_motor_speeds(self.max_speed * 0.6, -self.max_speed * 0.6)

            elif self.state == "FOLLOWING_TARGET":
                sensor_values = [s.getValue() for s in self.sensors]
                # Check for imminent collision
                if sensor_values[0] > 250 or sensor_values[7] > 250:
                    print(f"[{self.robot_name}] Obstacle detected! Aborting target and avoiding.")
                    self.state = "AVOIDING_OBSTACLE"
                    self.avoidance_step = 0
                    self.set_motor_speeds(0, 0)
                    continue

                if not self.current_target:
                    self.state = "SELECTING_GOAL"
                    continue
                
                target_vector = np.array(self.current_target) - self.position
                if np.linalg.norm(target_vector) < self.target_threshold:
                    print(f"[{self.robot_name}] Target reached.")
                    self.current_target = None
                    self.state = "SELECTING_GOAL"
                else: # Steer towards target with minor wall avoidance
                    target_angle = math.atan2(target_vector[1], target_vector[0])
                    angle_diff = target_angle - self.orientation
                    while angle_diff > np.pi: angle_diff -= 2 * np.pi
                    while angle_diff < -np.pi: angle_diff += 2 * np.pi
                    steer = np.clip(angle_diff * 1.5, -2.0, 2.0)
                    side_steer = (sensor_values[5] - sensor_values[2]) / 400.0 # Gentle nudge from side walls
                    final_steer = steer + side_steer
                    self.set_motor_speeds(self.max_speed - final_steer * 2.0, self.max_speed + final_steer * 2.0)

            elif self.state == "AVOIDING_OBSTACLE":
                # Execute a clean backup-and-turn maneuver
                if self.avoidance_step < 15: # Backup
                    self.set_motor_speeds(-self.max_speed * 0.6, -self.max_speed * 0.6)
                elif self.avoidance_step < 40: # Turn
                     self.set_motor_speeds(self.max_speed * 0.7, -self.max_speed * 0.7)
                else: # Maneuver complete, find a new goal
                    print(f"[{self.robot_name}] Avoidance complete. Finding new goal.")
                    self.state = "SELECTING_GOAL"
                    self.avoidance_step = 0
                self.avoidance_step += 1

if __name__ == "__main__":
    controller = EPuckController()
    controller.run()