import cv2
import numpy as np
import math
import os
import time

class MapVisualizer:
    def __init__(self, resolution=0.2, window_size=800):
        self.resolution = resolution
        self.window_size = window_size
        self.scale = window_size / 10.0  # 10m x 10m virtual world
        
        # Define colors
        self.OBSTACLE_COLOR = (0, 0, 0)       # Black
        self.FREE_SPACE_COLOR = (240, 240, 240) # Light gray
        self.PATH_COLOR = (0, 255, 0)         # Green
        self.FRONTIER_COLOR = (255, 165, 0)    # Orange
        
        # Add a class variable for the window name
        self.window_name = "Multi-Robot Map"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Add static map storage
        self.static_map = np.ones((window_size, window_size, 3), dtype=np.uint8) * 255
        self.robot_colors = {}  # Store unique colors for each robot
        self.next_color_idx = 0
        self.base_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 165, 0),  # Orange
            (128, 0, 128),  # Purple
            (0, 255, 255),  # Cyan
            (255, 0, 255)   # Magenta
        ]

        # Create output directory for maps
        self.output_dir = "robot_maps"
        os.makedirs(self.output_dir, exist_ok=True)

    def get_robot_color(self, robot_id):
        """Assign a unique color to each robot"""
        if robot_id not in self.robot_colors:
            color = self.base_colors[self.next_color_idx % len(self.base_colors)]
            self.robot_colors[robot_id] = color
            self.next_color_idx += 1
        return self.robot_colors[robot_id]

    def world_to_pixel(self, x, y):
        """Convert world coordinates to pixel coordinates"""
        px = int((x / self.resolution + self.window_size/2))
        py = int((-y / self.resolution + self.window_size/2))
        return px, py

    def update_display(self, robot_x, robot_y, heading, map_data, planned_path=None, 
                      other_robots=None, frontiers=None, robot_id="Robot"):
        """Update the map visualization"""
        # Update static map with new data
        for cell, value in map_data.items():
            x, y = cell
            px, py = self.world_to_pixel(x * self.resolution, y * self.resolution)
            if 0 <= px < self.window_size and 0 <= py < self.window_size:
                if value == 1:  # Obstacle
                    cv2.circle(self.static_map, (px, py), 2, self.OBSTACLE_COLOR, -1)
                else:  # Free space
                    cv2.circle(self.static_map, (px, py), 2, (240, 240, 240), -1)

        # Create a copy of the static map for this frame
        self.map_image = self.static_map.copy()

        # Draw current robot with its unique color
        robot_color = self.get_robot_color(robot_id)
        px, py = self.world_to_pixel(robot_x, robot_y)
        cv2.circle(self.map_image, (px, py), 8, robot_color, -1)
        
        # Draw robot heading
        end_x = px + int(15 * math.cos(heading))
        end_y = py - int(15 * math.sin(heading))
        cv2.line(self.map_image, (px, py), (end_x, end_y), robot_color, 2)

        # Draw other robots with their unique colors
        if other_robots:
            for other_id, info in other_robots.items():
                pos = info.get('position', {})
                if 'x' in pos and 'y' in pos:
                    other_color = self.get_robot_color(other_id)
                    px, py = self.world_to_pixel(pos['x'], pos['y'])
                    cv2.circle(self.map_image, (px, py), 8, other_color, -1)

        # Draw planned paths
        if planned_path:
            path_points = []
            for cell in planned_path:
                px, py = self.world_to_pixel(cell[0] * self.resolution, 
                                           cell[1] * self.resolution)
                path_points.append((px, py))
            if len(path_points) > 1:
                for i in range(len(path_points)-1):
                    cv2.line(self.map_image, path_points[i], path_points[i+1], 
                            self.PATH_COLOR, 2)

        # Add legend
        y_offset = 20
        cv2.putText(self.map_image, "Robots Present:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 20
        
        # Show all robot IDs and their colors in legend
        for rid, color in self.robot_colors.items():
            cv2.putText(self.map_image, f"{rid}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_offset += 20

        # Show the map
        cv2.imshow(self.window_name, self.map_image)
        cv2.waitKey(1)

    def save_map(self, base_filename=None):
        """Save the current map visualization with timestamp"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if base_filename:
            filename = f"{self.output_dir}/map_{timestamp}_{base_filename}.png"
        else:
            filename = f"{self.output_dir}/map_{timestamp}.png"
        cv2.imwrite(filename, self.map_image)
        print(f"Map saved to: {filename}")