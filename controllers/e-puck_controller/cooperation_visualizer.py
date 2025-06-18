# cooperation_visualizer.py
import cv2
import numpy as np
import time
from collections import deque

class CooperationVisualizer:
    """
    Creates a dedicated OpenCV window to visualize the cooperation and data 
    sharing between robots, rather than just showing the resulting map.
    """
    def __init__(self, robot_names):
        self.robot_names = sorted(robot_names)
        self.num_robots = len(self.robot_names)
        
        # --- Window and Canvas Setup ---
        self.window_name = "Cooperation & Data Sharing Dashboard"
        self.width, self.height = 800, 600
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # --- Data Storage ---
        self.robot_states = {name: {} for name in self.robot_names}
        self.comm_links = {}  # Tracks who is sending data to whom
        self.event_log = deque(maxlen=10) # Shows last 10 major events
        
        # --- Visualization Parameters ---
        self.colors = {
            "background": (15, 15, 15),
            "text": (240, 240, 240),
            "header": (100, 200, 255),
            "link": (0, 255, 100),
            "detection": (50, 100, 255),
            "state_exploring": (100, 255, 100),
            "state_avoiding": (255, 150, 50),
            "state_goal_select": (255, 255, 100),
        }
        self.robot_node_positions = self._calculate_node_positions()
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def _calculate_node_positions(self):
        """Calculates circular positions for robot nodes."""
        positions = {}
        center_x, center_y = self.width // 2, 220
        radius = 150
        for i, name in enumerate(self.robot_names):
            angle = (2 * np.pi / self.num_robots) * i - (np.pi / 2)
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            positions[name] = (x, y)
        return positions

    def update_data(self, robot_name, state_data):
        """Receives a comprehensive update for a single robot."""
        if robot_name in self.robot_states:
            self.robot_states[robot_name] = state_data

            # If this update indicates a sync, log the communication
            if state_data.get('last_sync_time'):
                self.log_communication_event(robot_name, "all")

            # If there are new detections, log them as events
            if state_data.get('new_detections'):
                 for det in state_data['new_detections']:
                    self.log_event(f"Robot '{robot_name}' detected a {det['class']}")

    def log_communication_event(self, from_robot, to_robot):
        """Logs a communication event to draw a temporary link."""
        # Key identifies the link, timestamp is for fade-out effect
        self.comm_links[f"{from_robot}->{to_robot}"] = time.time()

    def log_event(self, text):
        """Adds a new event to the scrolling log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.event_log.append(f"[{timestamp}] {text}")

    def render(self):
        """Draws the entire dashboard canvas."""
        # 1. Clear canvas
        self.canvas[:] = self.colors["background"]
        
        # 2. Draw Header
        cv2.putText(self.canvas, "Robot Cooperation Dashboard", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors["header"], 2)
        
        # 3. Draw Communication Links
        current_time = time.time()
        # Use a list to avoid changing dict during iteration
        for link_key, timestamp in list(self.comm_links.items()):
            if current_time - timestamp > 1.0: # Link fades after 1 second
                del self.comm_links[link_key]
                continue
            
            from_robot, _ = link_key.split('->')
            start_pos = self.robot_node_positions[from_robot]
            # Draw a line from the sender to the center, representing a broadcast
            cv2.line(self.canvas, start_pos, (self.width // 2, 220), self.colors["link"], 2)

        # 4. Draw Robot Nodes and Statuses
        for name, pos in self.robot_node_positions.items():
            state_info = self.robot_states.get(name, {})
            current_state = state_info.get('state', 'UNKNOWN')
            
            # Determine color based on state
            color = self.colors["text"]
            if "FOLLOWING" in current_state: color = self.colors["state_exploring"]
            elif "AVOIDING" in current_state: color = self.colors["state_avoiding"]
            elif "SELECTING" in current_state: color = self.colors["state_goal_select"]
                
            # Draw node and name
            cv2.circle(self.canvas, pos, 30, color, -1)
            cv2.putText(self.canvas, name, (pos[0]-25, pos[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            
            # Display status text
            cv2.putText(self.canvas, f"State: {current_state}", (pos[0]+40, pos[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["text"], 1)
            target = state_info.get('target')
            target_str = f"({target[0]:.1f}, {target[1]:.1f})" if target else "None"
            cv2.putText(self.canvas, f"Target: {target_str}", (pos[0]+40, pos[1]+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["text"], 1)

        # 5. Draw Event Log
        cv2.line(self.canvas, (20, self.height - 180), (self.width - 20, self.height - 180), self.colors["header"], 1)
        cv2.putText(self.canvas, "Shared Event Log", (20, self.height - 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors["header"], 2)
        for i, event_text in enumerate(reversed(self.event_log)):
            y_pos = self.height - 150 + (i * 20)
            cv2.putText(self.canvas, event_text, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["detection"], 1)

        # 6. Display the final image
        cv2.imshow(self.window_name, self.canvas)
        cv2.waitKey(1)