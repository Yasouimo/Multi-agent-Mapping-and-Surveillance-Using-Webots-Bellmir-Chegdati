import numpy as np
import pickle
import os
import glob
import time
import cv2

class CooperativeMapping:
    def __init__(self, robot_name="", world_size=(2.0, 2.0), grid_size=0.05):
        self.robot_name = robot_name
        self.world_size, self.grid_size = world_size, grid_size
        self.grid_width = int(world_size[0] / grid_size)
        self.grid_height = int(world_size[1] / grid_size)
        self.UNKNOWN, self.FREE, self.OCCUPIED = 128, 255, 0
        
        self.data_dir = "robot_data"
        if not os.path.exists(self.data_dir): os.makedirs(self.data_dir)
        self.my_data_file = os.path.join(self.data_dir, f"data_{self.robot_name}.pkl")
        
        self.grid_map = np.full((self.grid_height, self.grid_width), self.UNKNOWN, dtype=np.uint8)
        self.robot_states = {}
        
        self.viz_enabled = (self.robot_name == "e-puck")
        if self.viz_enabled:
            cv2.namedWindow("Cooperative Map", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Cooperative Map", 600, 600)

    def world_to_grid(self, world_pos):
        grid_x = int((world_pos[0] + self.world_size[0] / 2) / self.grid_size)
        grid_y = int((world_pos[1] + self.world_size[1] / 2) / self.grid_size)
        return np.clip(grid_y, 0, self.grid_height - 1), np.clip(grid_x, 0, self.grid_width - 1)

    def grid_to_world(self, grid_pos):
        world_x = (grid_pos[1] + 0.5) * self.grid_size - self.world_size[0] / 2
        world_y = (grid_pos[0] + 0.5) * self.grid_size - self.world_size[1] / 2
        return world_x, world_y

    def update_map_from_sensors(self, robot_pos, robot_orientation, sensor_readings, max_range=0.12, obs_threshold=100):
        robot_grid = self.world_to_grid(robot_pos)
        self.grid_map[robot_grid] = self.FREE
        sensor_angles = robot_orientation + np.deg2rad([90, 45, 0, -45, -90, -135, 180, 135])
        for angle, reading in zip(sensor_angles, sensor_readings):
            distance = max_range * (1 - min(reading, 1000) / 1000.0)
            for step in np.linspace(0, distance, int(distance / self.grid_size) + 1):
                point_grid = self.world_to_grid(robot_pos + np.array([step * np.cos(angle), step * np.sin(angle)]))
                if self.grid_map[point_grid] == self.UNKNOWN: self.grid_map[point_grid] = self.FREE
            if reading > obs_threshold:
                obs_grid = self.world_to_grid(robot_pos + np.array([distance * np.cos(angle), distance * np.sin(angle)]))
                self.grid_map[obs_grid] = self.OCCUPIED

    def find_exploration_frontiers(self):
        free_mask = (self.grid_map == self.FREE).astype(np.uint8)
        unknown_mask = (self.grid_map == self.UNKNOWN).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        adjacent_to_unknown = cv2.dilate(unknown_mask, kernel)
        frontiers_mask = (free_mask == 1) & (adjacent_to_unknown == 1)
        frontier_coords = np.argwhere(frontiers_mask)
        if len(frontier_coords) < 10: return [tuple(c) for c in frontier_coords]
        num_clusters = max(8, min(25, len(frontier_coords) // 50))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(frontier_coords.astype(np.float32), num_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        return [tuple(c.astype(int)) for c in centers]

    def assign_exploration_target(self, my_pos, my_robot_name):
        self.load_all_robot_data()
        frontiers = self.find_exploration_frontiers()
        if not frontiers: return None

        my_grid_pos = np.array(self.world_to_grid(my_pos))
        other_robots_pos = [self.world_to_grid(s['position']) for r, s in self.robot_states.items() if r != my_robot_name]
        
        best_frontier = None
        max_score = -float('inf')

        for f_grid in frontiers:
            f_grid = np.array(f_grid)
            dist_to_self = np.linalg.norm(f_grid - my_grid_pos)
            
            if dist_to_self < 5: continue

            score = dist_to_self
            
            for r_pos in other_robots_pos:
                if np.linalg.norm(f_grid - np.array(r_pos)) < dist_to_self:
                    score *= 0.1

            y, x = int(f_grid[0]), int(f_grid[1])
            region_size = 5
            y_min, y_max = max(0, y - region_size), min(self.grid_height, y + region_size)
            x_min, x_max = max(0, x - region_size), min(self.grid_width, x + region_size)
            local_region = self.grid_map[y_min:y_max, x_min:x_max]
            
            openness = np.sum(local_region == self.FREE)
            score *= (1 + openness / 25.0)

            if score > max_score:
                max_score = score
                best_frontier = tuple(f_grid)
        
        return self.grid_to_world(best_frontier) if best_frontier is not None else None

    def sync_data(self, my_pos, my_target, my_robot_name):
        my_data = {
            'map_update': self.grid_map, 'position': list(my_pos),
            'target': list(my_target) if my_target is not None else None,
            'timestamp': time.time()
        }
        with open(self.my_data_file, 'wb') as f: pickle.dump(my_data, f)
        self.load_all_robot_data()
        if self.viz_enabled: self.display_map()

    def load_all_robot_data(self):
        all_data_files = glob.glob(os.path.join(self.data_dir, "data_*.pkl"))
        for f_path in all_data_files:
            try:
                robot_id = os.path.basename(f_path).split('_')[1].split('.')[0]
                with open(f_path, 'rb') as f: data = pickle.load(f)
                self.robot_states[robot_id] = data
                other_map = data['map_update']
                update_mask = (self.grid_map == self.UNKNOWN) & (other_map != self.UNKNOWN)
                self.grid_map[update_mask] = other_map[update_mask]
                obs_mask = other_map == self.OCCUPIED
                self.grid_map[obs_mask] = self.OCCUPIED
            except Exception: pass

    def display_map(self):
        display = cv2.cvtColor(self.grid_map, cv2.COLOR_GRAY2BGR)
        frontiers = self.find_exploration_frontiers()
        for r, c in frontiers: display[r, c] = [0, 255, 255]
        for name, state in self.robot_states.items():
            color = [255, 0, 0] if name == self.robot_name else [0, 165, 255]
            pos_grid = self.world_to_grid(state['position'])
            cv2.circle(display, (pos_grid[1], pos_grid[0]), 3, color, -1)
            if state.get('target') and state['target'] is not None:
                try:
                    target_grid = self.world_to_grid(state['target'])
                    cv2.line(display, (pos_grid[1], pos_grid[0]), (target_grid[1], target_grid[0]), color, 1)
                    cv2.circle(display, (target_grid[1], target_grid[0]), 3, [0, 0, 255], -1)
                except (ValueError, TypeError):
                    pass
        display = cv2.resize(display, (600, 600), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Cooperative Map", display)
        cv2.waitKey(1)