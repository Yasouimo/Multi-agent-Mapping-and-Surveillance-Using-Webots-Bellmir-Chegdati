from controller import Receiver, Emitter
import struct
import json
import math
import os
import csv
import time
from datetime import datetime

class RobotCommunicator:
    """
    Handles inter-robot communication for e-puck robots using Webots' emitter and receiver devices.
    Allows robots to exchange position data, status updates, and coordination commands.
    """
    def __init__(self, robot, channel=1, max_range=10.0):
        """
        Initialize the robot communicator.
        
        Args:
            robot: The Webots Robot instance
            channel: Communication channel (default: 1)
            max_range: Maximum communication range in meters (default: 10.0)
        """
        # Add these new class variables in the __init__ method
        self.first_detections = {}  # Track first detections by object ID
        self.alarm_activated = False  # Track if alarm has been activated
        self.robot = robot
        self.channel = channel
        self.max_range = max_range
        self.robot_id = None  # Will be set on first broadcast
        
        # Get robot name or generate a simple ID
        self.robot_name = robot.getName() or f"EPuck_{id(robot) % 1000}"
        
        # Initialize emitter
        try:
            self.emitter = robot.getDevice("emitter")
            if self.emitter:
                self.emitter.setChannel(channel)
                self.emitter.setRange(max_range)
                print(f"Emitter initialized on channel {channel} with range {max_range}m")
            else:
                print("Warning: Emitter device not found")
                self.emitter = None
        except Exception as e:
            print(f"Failed to initialize emitter: {e}")
            self.emitter = None
            
        # Initialize receiver
        try:
            self.receiver = robot.getDevice("receiver")
            if self.receiver:
                self.receiver.enable(int(robot.getBasicTimeStep()))
                self.receiver.setChannel(channel)
                print(f"Receiver initialized on channel {channel}")
            else:
                print("Warning: Receiver device not found")
                self.receiver = None
        except Exception as e:
            print(f"Failed to initialize receiver: {e}")
            self.receiver = None
            
        # Message buffer for received messages
        self.received_messages = []
        
        # Set up a unique robot ID based on timestamp
        import time
        self.robot_id = f"epuck_{int(time.time() * 1000) % 10000}"
        print(f"Robot ID assigned: {self.robot_id}")
        
        # Last known positions of other robots
        self.robot_positions = {}
        
        # Last broadcast time
        self.last_broadcast_time = 0
        
        # Track last cat detection time for cooperative detection
        self.last_cat_detection_time = 0
        
        # Known detections from all robots
        self.all_detections = {}
        
        # Track detections and their order
        self.detections_log = "robot_detections.csv"
        self.known_detections = {}  # Format: {object_type: [robot_names_in_order]}
        
        # Initialize CSV file
        self.init_csv_file()
        
    def init_csv_file(self):
        """Create CSV file with updated headers"""
        if not os.path.exists(self.detections_log):
            with open(self.detections_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Time', 'Robot', 'Object', 'Detection_Order', 'Position', 'First_Detector'])
            
    def log_message(self, message, receiver_id="broadcast"):
        """Log message to CSV file with improved formatting"""
        pass  # Removed CSV logging functionality
        
    def log_detection(self, object_type, position):
        """Log detection with order and ancestor tracking"""
        try:
            # Skip logging obstacles
            if (object_type == "Obstacle"):
                return False

            if object_type not in self.known_detections:
                self.known_detections[object_type] = []
                
            # Check if this robot already detected this object
            if self.robot_name not in self.known_detections[object_type]:
                self.known_detections[object_type].append(self.robot_name)
                detection_order = len(self.known_detections[object_type])
                first_detector = self.known_detections[object_type][0] if detection_order > 1 else "None"
            else:
                detection_order = "Repeat"
                first_detector = self.known_detections[object_type][0]
                
            # Log to CSV
            with open(self.detections_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%H:%M:%S"),
                    self.robot_name,
                    object_type,
                    detection_order,
                    f"({position[0]:.2f}, {position[1]:.2f})",
                    first_detector if detection_order == "Repeat" else "First"
                ])
                
            return detection_order == 1

        except Exception as e:
            print(f"Failed to log detection: {e}")
            return False
        
    def broadcast_position(self, x, y, heading=0.0, status="exploring"):
        """
        Broadcast this robot's position and status to other robots.
        
        Args:
            x: X coordinate of the robot
            y: Y coordinate of the robot
            heading: Heading angle in radians
            status: Robot status (exploring, detected_object, etc.)
        
        Returns:
            True if broadcast successful, False otherwise
        """
        if not self.emitter:
            return False
            
        try:
            # Create message
            message = {
                "type": "position",
                "robot_id": self.robot_id,
                "position": {
                    "x": float(x),
                    "y": float(y),
                    "heading": float(heading)
                },
                "status": status,
                "timestamp": self.robot.getTime()
            }
            
            # Convert to JSON and send
            json_message = json.dumps(message)
            self.emitter.send(json_message.encode('utf-8'))
            
            # Log the message
            self.log_message(message)
            
            self.last_broadcast_time = self.robot.getTime()
            return True
        except Exception as e:
            print(f"Failed to broadcast position: {e}")
            return False
    
    def broadcast_detection(self, detection_type, confidence, position):
        """Broadcast detection to other robots"""
        try:
            # Ensure position is properly formatted
            location = {
                "x": float(position[0]) if isinstance(position, (list, tuple)) else float(position["x"]),
                "y": float(position[1]) if isinstance(position, (list, tuple)) else float(position["y"])
            }

            message = {
                "from": self.robot_name,
                "type": "detection",
                "object": detection_type,
                "confidence": float(confidence),
                "location": location,
                "timestamp": self.robot.getTime()
            }

            if self.emitter:
                self.emitter.send(json.dumps(message).encode('utf-8'))
                
            # Log detection
            detection_order = self.log_detection(detection_type, (location["x"], location["y"]))
            return detection_order == 1  # Return True if first detection
                
        except Exception as e:
            print(f"Failed to broadcast detection: {e}")
            return False
    
    def broadcast_command(self, command_type, params=None):
        """
        Broadcast a command to other robots.
        
        Args:
            command_type: Type of command (e.g., "investigate", "avoid", "regroup")
            params: Optional dictionary with command parameters
        
        Returns:
            True if broadcast successful, False otherwise
        """
        if not self.emitter:
            return False
            
        try:
            # Create message
            message = {
                "type": "command",
                "robot_id": self.robot_id,
                "command": {
                    "action": command_type
                },
                "timestamp": self.robot.getTime()
            }
            
            # Add parameters if provided
            if params:
                message["command"]["params"] = params
                
            # Convert to JSON and send
            json_message = json.dumps(message)
            self.emitter.send(json_message.encode('utf-8'))
            
            # Log the message
            self.log_message(message)
            
            return True
        except Exception as e:
            print(f"Failed to broadcast command: {e}")
            return False
    
    def check_for_messages(self):
        """
        Check for and process any received messages.
        
        Returns:
            List of new messages received
        """
        if not self.receiver:
            return []
            
        new_messages = []
        
        # Process all available messages
        while self.receiver.getQueueLength() > 0:
            try:
                # Get the message as string - Using getString instead of getData
                message_string = self.receiver.getString()
                
                # Parse the JSON message (no need to decode as getString returns a string)
                message = json.loads(message_string)
                
                # Skip our own messages
                if message.get("robot_id") == self.robot_id:
                    self.receiver.nextPacket()
                    continue
                    
                # Log the received message
                self.log_message(message, receiver_id=self.robot_id)
                
                # Process different message types
                if message["type"] == "position":
                    # Update our knowledge of other robots' positions
                    self.robot_positions[message["robot_id"]] = {
                        "position": message["position"],
                        "status": message["status"],
                        "timestamp": message["timestamp"]
                    }
                
                elif message["type"] == "detection":
                    # Store the detection information
                    robot_id = message["robot_id"]
                    detection_type = message["detection"]["object_type"]
                    confidence = message["detection"]["confidence"]
                    timestamp = message["timestamp"]
                    
                    if robot_id not in self.all_detections:
                        self.all_detections[robot_id] = []
                        
                    self.all_detections[robot_id].append({
                        "type": detection_type,
                        "confidence": confidence,
                        "timestamp": timestamp,
                        "location": message["detection"].get("location")
                    })
                    
                    # If it's a cat detection, update the cooperative detection time
                    if detection_type == "Cat":
                        self.last_cat_detection_time = timestamp
                
                # Add to our list of new messages
                new_messages.append(message)
                
                # Advance to the next packet
                self.receiver.nextPacket()
                
            except Exception as e:
                print(f"Error processing received message: {e}")
                self.receiver.nextPacket()  # Skip the problematic message
        
        return new_messages
    
    def check_messages(self):
        """Process incoming messages from other robots"""
        if not self.receiver:
            return []
            
        messages = []
        while self.receiver.getQueueLength() > 0:
            try:
                message = json.loads(self.receiver.getString())
                if message.get("from", "") != self.robot_name:  # Safe get for "from" field
                    if message.get("type") == "detection":  # Safe get for "type" field
                        object_type = message.get("object")
                        if object_type and object_type != "Obstacle":  # Only process non-obstacle detections
                            if object_type not in self.known_detections:
                                self.known_detections[object_type] = []
                            if message["from"] not in self.known_detections[object_type]:
                                self.known_detections[object_type].append(message["from"])
                            messages.append(message)
            except Exception as e:
                print(f"Error processing message: {e}")
            finally:
                self.receiver.nextPacket()
        return messages
    
    def get_nearby_robots(self, max_distance=None):
        """
        Get a list of nearby robots based on their last known positions.
        
        Args:
            max_distance: Maximum distance to consider (defaults to emitter range)
            
        Returns:
            Dictionary of nearby robot IDs and their information
        """
        if max_distance is None:
            max_distance = self.max_range
            
        # Get current time
        current_time = self.robot.getTime()
        
        # Filter for robots that are nearby and recently updated
        nearby_robots = {}
        for robot_id, info in self.robot_positions.items():
            # Skip outdated positions (older than 30 seconds)
            if current_time - info["timestamp"] > 30:
                continue
                
            # Calculate distance (if we know our position)
            # This requires the controller to provide our position
            if "self_position" in self.__dict__:
                x1, y1 = self.self_position["x"], self.self_position["y"]
                x2, y2 = info["position"]["x"], info["position"]["y"]
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                if distance <= max_distance:
                    info["distance"] = distance
                    nearby_robots[robot_id] = info
            else:
                # If we don't know our position, include all robots
                nearby_robots[robot_id] = info
                
        return nearby_robots
    
    def should_trigger_alarm(self):
        """
        Determine if this robot should trigger an alarm based on cooperative logic.
        
        Returns:
            Boolean indicating if alarm should be triggered
        """
        current_time = self.robot.getTime()
        # Only trigger alarm if no other cat detection in the last 60 seconds
        return (current_time - self.last_cat_detection_time) > 60.0
        
    def update(self, robot_x=None, robot_y=None, heading=None, status="exploring", broadcast_interval=2.0):
        """
        Update routine to be called regularly from the main controller.
        Checks for incoming messages and broadcasts position if needed.
        
        Args:
            robot_x: Current X position
            robot_y: Current Y position
            heading: Current heading angle in radians
            status: Current robot status
            broadcast_interval: How often to broadcast position (seconds)
            
        Returns:
            List of new messages
        """
        # Store our position if provided
        if robot_x is not None and robot_y is not None:
            self.self_position = {"x": robot_x, "y": robot_y}
            if heading is not None:
                self.self_position["heading"] = heading
        
        # Check for received messages
        new_messages = self.check_for_messages()
        
        # Broadcast our position periodically
        current_time = self.robot.getTime()
        if robot_x is not None and robot_y is not None:
            if current_time - self.last_broadcast_time >= broadcast_interval:
                self.broadcast_position(robot_x, robot_y, heading or 0.0, status)
                
        return new_messages
    
    def handle_detection_alert(self, detection_type, confidence, x=None, y=None):
        """Handle and broadcast a detection alert with improved alarm control"""
        location = None
        if x is not None and y is not None:
            location = {"x": x, "y": y}
            
        # For Cat detections, check cooperative alarm logic
        if detection_type == "Cat" and confidence > 0.5:
            current_time = self.robot.getTime()
            
            # Only trigger if we haven't detected a cat recently and alarm isn't already activated
            if not self.alarm_activated and (current_time - self.last_cat_detection_time) > 60.0:
                self.alarm_activated = True
                self.last_cat_detection_time = current_time
                success = self.broadcast_detection(detection_type, confidence, location)
                
                if success:
                    self.broadcast_command("investigate", {
                        "target": detection_type,
                        "priority": "high",
                        "location": location
                    })
                return success
            else:
                print("🐱 Cat detected but alarm suppressed (already active or too recent)")
                return False
        
        # For non-cat detections, just broadcast normally
        return self.broadcast_detection(detection_type, confidence, location)
    
    def get_recent_detections(self, detection_type=None, max_age=30):
        """
        Get recent detections of the specified type from all robots.
        
        Args:
            detection_type: Optional type of detection to filter for
            max_age: Maximum age of detections in seconds
            
        Returns:
            List of recent detections
        """
        current_time = self.robot.getTime()
        recent_detections = []
        
        for robot_id, detections in self.all_detections.items():
            for detection in detections:
                # Skip if too old
                if current_time - detection["timestamp"] > max_age:
                    continue
                    
                # Skip if not matching type (if specified)
                if detection_type and detection["type"] != detection_type:
                    continue
                    
                # Add robot ID to the detection info
                detection_info = detection.copy()
                detection_info["robot_id"] = robot_id
                recent_detections.append(detection_info)
                
        return recent_detections