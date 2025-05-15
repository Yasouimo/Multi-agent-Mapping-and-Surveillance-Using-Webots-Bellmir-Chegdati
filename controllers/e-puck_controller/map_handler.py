import numpy as np
from heapq import heappush, heappop
import math
import time

class MapHandler:
    def __init__(self, resolution=0.2):
        """
        Initialize the map handler.
        
        Args:
            resolution: Map grid resolution in meters
        """
        self.resolution = resolution
        self.map = {}  # Occupancy grid as sparse dictionary
        self.robot_paths = {}  # Store other robots' planned paths
        self.explored_areas = set()  # Track explored grid cells
        self.coordination_stats = {
            'maps_received': 0,
            'cells_shared': 0,
            'paths_coordinated': 0
        }
    
    def update_map(self, x, y, is_obstacle):
        """
        Update map with new observation.
        
        Args:
            x, y: Real-world coordinates
            is_obstacle: Boolean indicating if obstacle detected
        """
        # Convert to grid coordinates
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)
        cell = (grid_x, grid_y)
        
        # Update occupancy
        if is_obstacle:
            self.map[cell] = 1  # Obstacle
        else:
            self.map[cell] = 0  # Free space
            self.explored_areas.add(cell)
    
    def merge_map(self, other_map):
        """Merge received map data from other robots"""
        cells_before = len(self.map)
        
        for cell_str, value in other_map.items():
            try:
                x, y = map(int, cell_str.split(','))
                cell = (x, y)
                if cell not in self.map or value == 1:
                    self.map[cell] = value
                    if value == 0:
                        self.explored_areas.add(cell)
            except Exception as e:
                print(f"Error processing map cell {cell_str}: {e}")
        
        cells_added = len(self.map) - cells_before
        self.coordination_stats['maps_received'] += 1
        self.coordination_stats['cells_shared'] += cells_added
        
        print(f"📊 Map Merge Statistics:")
        print(f"  - New cells added: {cells_added}")
        print(f"  - Total maps received: {self.coordination_stats['maps_received']}")
        print(f"  - Total cells shared: {self.coordination_stats['cells_shared']}")
        
    def get_coordination_stats(self):
        """Return current coordination statistics"""
        return self.coordination_stats
    
    def find_unexplored_frontier(self, current_x, current_y):
        """Find nearest unexplored area"""
        current_cell = (int(current_x / self.resolution), 
                       int(current_y / self.resolution))
        
        # Use A* to find nearest unexplored cell
        frontier = []
        heappush(frontier, (0, current_cell))
        came_from = {current_cell: None}
        cost_so_far = {current_cell: 0}
        
        while frontier:
            _, current = heappop(frontier)
            
            # Check if current cell is at frontier of explored area
            if self._is_frontier(current):
                return self._reconstruct_path(came_from, current)
            
            # Check neighboring cells
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                next_cell = (current[0] + dx, current[1] + dy)
                
                # Skip if obstacle
                if self.map.get(next_cell, 0) == 1:
                    continue
                
                new_cost = cost_so_far[current] + 1
                if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                    cost_so_far[next_cell] = new_cost
                    priority = new_cost + self._heuristic(next_cell, current_cell)
                    heappush(frontier, (priority, next_cell))
                    came_from[next_cell] = current
        
        return None  # No unexplored areas found
    
    def _is_frontier(self, cell):
        """Check if cell is at frontier of explored area"""
        if cell in self.explored_areas:
            return False
            
        # Check if cell has adjacent explored area
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            neighbor = (cell[0] + dx, cell[1] + dy)
            if neighbor in self.explored_areas:
                return True
        return False
    
    def _heuristic(self, a, b):
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _reconstruct_path(self, came_from, goal):
        """Reconstruct path from A* search"""
        current = goal
        path = []
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path
    
    def update_robot_path(self, robot_id, path):
        """Store other robot's planned path"""
        self.robot_paths[robot_id] = {
            'path': path,
            'timestamp': time.time()
        }
    
    def get_path_to_target(self, start_x, start_y, target_x, target_y):
        """Find path to specific target using A*"""
        start = (int(start_x / self.resolution), 
                int(start_y / self.resolution))
        goal = (int(target_x / self.resolution), 
               int(target_y / self.resolution))
        
        frontier = []
        heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            _, current = heappop(frontier)
            
            if current == goal:
                return self._reconstruct_path(came_from, goal)
            
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                next_cell = (current[0] + dx, current[1] + dy)
                
                # Skip if obstacle
                if self.map.get(next_cell, 0) == 1:
                    continue
                
                new_cost = cost_so_far[current] + 1
                if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                    cost_so_far[next_cell] = new_cost
                    priority = new_cost + self._heuristic(next_cell, goal)
                    heappush(frontier, (priority, next_cell))
                    came_from[next_cell] = current
        
        return None  # No path found