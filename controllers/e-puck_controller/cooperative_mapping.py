# cooperative_mapping.py
import numpy as np
import pickle
import os
import glob
import time
import cv2 

# For creating high-quality, savable plots
import matplotlib.pyplot as plt


class CooperativeMapping:
    def __init__(self, robot_name="", world_size=(2.0, 2.0), grid_size=0.05):
        self.robot_name = robot_name
        self.world_size, self.grid_size = world_size, grid_size
        self.grid_width = int(world_size[0] / grid_size)
        self.grid_height = int(world_size[1] / grid_size)
        self.UNKNOWN, self.FREE, self.OCCUPIED = 128, 255, 0
        
        self.data_dir = "robot_data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.my_data_file = os.path.join(self.data_dir, f"data_{self.robot_name}.pkl")
        
        self.grid_map = np.full((self.grid_height, self.grid_width), self.UNKNOWN, dtype=np.uint8)
        self.robot_states = {}
        
        self.viz_enabled = (self.robot_name == "e-puck")
        if self.viz_enabled:
            # Create a directory to store the dynamic map frames
            self.frames_dir = "map_frames"
            os.makedirs(self.frames_dir, exist_ok=True)
            self.frame_count = 0
            # Setup for the live CV2 window
            cv2.namedWindow("Cooperative Map", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Cooperative Map", 600, 600)

    def save_map_as_image(self):
        """
        Saves the current state of the map and robots as a high-quality PNG image.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        map_extent = [-self.world_size[0]/2, self.world_size[0]/2, -self.world_size[1]/2, self.world_size[1]/2]
        im = ax.imshow(self.grid_map, cmap='RdBu_r', vmin=0, vmax=255, extent=map_extent, origin='lower')
        
        fig.colorbar(im, ax=ax, label="Map Value (0=Occupied, 255=Free)")

        for name, state in self.robot_states.items():
            try:
                pos = state['position']
                target = state.get('target')
                color = 'green' if name == self.robot_name else 'orange'
                ax.scatter(pos[0], pos[1], c=color, s=100, edgecolors='black', zorder=3, label=f'Robot: {name}')
                if target and target is not None:
                    ax.scatter(target[0], target[1], c='red', marker='x', s=100, zorder=3, label=f'Target: {name}')
                    ax.plot([pos[0], target[0]], [pos[1], target[1]], 'r-', zorder=2)
            except (KeyError, TypeError):
                continue

        ax.set_xlim(-self.world_size[0]/2, self.world_size[0]/2)
        ax.set_ylim(-self.world_size[1]/2, self.world_size[1]/2)
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.set_title(f"Cooperative Map | Iteration: {self.frame_count} | Robots: {len(self.robot_states)}")
        ax.grid(True, linestyle='--', alpha=0.5)

        filepath = os.path.join(self.frames_dir, f"frame_{self.frame_count:05d}.png")
        plt.savefig(filepath, dpi=100)
        plt.close(fig)

    def world_to_grid(self, world_pos):
        grid_x = int((world_pos[0] + self.world_size[0] / 2) / self.grid_size)
        grid_y = int((world_pos[1] + self.world_size[1] / 2) / self.grid_size)
        return np.clip(grid_y, 0, self.grid_height - 1), np.clip(grid_x, 0, self.grid_width - 1)

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
    
    def sync_data(self, my_pos, my_target, my_robot_name, current_state, new_detections):
        self.robot_states[my_robot_name] = {'position': my_pos, 'target': my_target}
        my_data = {'map_update': self.grid_map, 'position': list(my_pos), 'target': list(my_target) if my_target is not None else None, 'state': current_state, 'new_detections': new_detections, 'timestamp': time.time()}
        with open(self.my_data_file, 'wb') as f: pickle.dump(my_data, f)
        
        self.load_all_robot_data()
        
        if self.viz_enabled:
            self.display_map()
            self.save_map_as_image()
            self.frame_count += 1

    def load_all_robot_data(self):
        all_data_files = glob.glob(os.path.join(self.data_dir, "data_*.pkl"))
        for f_path in all_data_files:
            try:
                if time.time() - os.path.getmtime(f_path) > 5.0: continue
                robot_id = os.path.basename(f_path).split('_')[1].split('.')[0]
                with open(f_path, 'rb') as f: data = pickle.load(f)
                
                self.robot_states[robot_id] = data 
                
                if robot_id == self.robot_name: continue
                
                # Merge the maps
                received_map = data['map_update']
                update_mask = (self.grid_map == self.UNKNOWN) & (received_map != self.UNKNOWN)
                self.grid_map[update_mask] = received_map[update_mask]
                obs_mask = received_map == self.OCCUPIED
                self.grid_map[obs_mask] = self.OCCUPIED

            except Exception: pass

    def display_map(self):
        display = cv2.cvtColor(self.grid_map, cv2.COLOR_GRAY2BGR)
        for name, state in self.robot_states.items():
            color = [0, 255, 0] if name == self.robot_name else [0, 165, 255]
            try:
                pos_grid = self.world_to_grid(state['position'])
                cv2.circle(display, (pos_grid[1], pos_grid[0]), 5, color, -1)
            except (KeyError, ValueError, TypeError): continue
        display = cv2.resize(display, (600, 600), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Cooperative Map", display)
        cv2.waitKey(1)