from controller import Robot, Camera, Speaker
import numpy as np
import os
import cv2
from datetime import datetime
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, robot):
        # Initialize robot components
        self.robot = robot
        self.time_step = int(robot.getBasicTimeStep())
        
        # Initialize camera
        self.camera = robot.getDevice("camera")
        if self.camera:
            self.camera.enable(self.time_step)
            self.camera_width = self.camera.getWidth()
            self.camera_height = self.camera.getHeight()
            print(f"Camera initialized: {self.camera_width}x{self.camera_height}")
        else:
            print("Warning: Camera not found")
            self.camera_width = 0
            self.camera_height = 0
        
        # Get speaker
        try:
            self.speaker = robot.getDevice("speaker")
            if self.speaker:
                print("Speaker initialized")
        except Exception as e:
            print(f"Speaker not available: {e}")
            self.speaker = None
        
        # Create detection folder
        self.detection_folder = "detections"
        os.makedirs(self.detection_folder, exist_ok=True)
        
        # Updated classes to match your YOLO model's class names
        self.target_classes = ['CardboardBox', 'Cat', 'OilBarrel', 'PlasticCrate', 'WoodenBox']
        
        # Alarm-triggering classes - only Cat from your class list
        self.alarm_classes = ["Cat"]
        
        # Alarm sound file path
        # Make sure this file exists in your Webots project directory
        self.alarm_sound_file = "mixkit-classic-short-alarm-993.wav"
        
        # Detection frequency
        self.detection_interval = 100  # Run detection every 100 steps
        
        # Load model lazily
        self.model = None
        
        print("Object detector initialized with classes:", self.target_classes)
    
    def play_alarm(self):
        """Play alarm for cat or dog detection"""
        try:
            if self.speaker:
                # Try different ways to play sound based on Webots version
                try:
                    # Newer Webots API
                    self.speaker.playSound(self.speaker, self.alarm_sound_file, 1.0, 1.0, 0, 0)
                    print("⚠️ ALARM: Animal detected! Sound played ⚠️")
                except Exception as e1:
                    try:
                        # Alternative API
                        self.speaker.playSound(self.alarm_sound_file, 1.0)
                        print("⚠️ ALARM: Animal detected! Sound played (alternative method) ⚠️")
                    except Exception as e2:
                        print(f"⚠️ ALARM: Animal detected! (Speaker error: {e1}, {e2}) ⚠️")
            else:
                # No speaker but still notify
                print("⚠️ ALARM: Animal detected! (No speaker available) ⚠️")
        except Exception as e:
            print(f"⚠️ ALARM: Animal detected! (Error playing sound: {e}) ⚠️")
    
    def process_frame(self):
        """Process camera frame and save detected objects as images"""
        # Skip if no camera
        if not self.camera:
            print("No camera available for detection")
            return False
        
        # Load model if needed
        if self.model is None:
            try:
                # Make sure the best.pt file is in your project directory
                model_path = "best.pt"
                if not os.path.exists(model_path):
                    print(f"Error: Model file {model_path} not found!")
                    return False
                
                self.model = YOLO(model_path)
                print(f"YOLO model loaded successfully with classes: {self.model.names}")
            except Exception as e:
                print(f"Failed to load YOLOv8 model: {e}")
                return False
        
        # Get image from camera
        try:
            image = self.camera.getImage()
            if not image:
                print("Failed to get image from camera")
                return False
            
            # Convert to OpenCV format - with error checking
            try:
                img = np.frombuffer(image, np.uint8).reshape((self.camera_height, self.camera_width, 4))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            except Exception as e:
                print(f"Error converting camera image: {e}")
                return False
                
            # Run detection with explicit error handling
            try:
                results = self.model(img, conf=0.25, verbose=False)
            except Exception as e:
                print(f"YOLO inference error: {e}")
                return False
            
            # Check for detections
            alarm_triggered = False
            detections_found = False
            
            if results and len(results) > 0:
                # Get result data
                result = results[0]
                if len(result.boxes) > 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    detections_found = True
                    
                    # Process each detection
                    for i, box in enumerate(result.boxes):
                        # Get class info
                        cls_id = int(box.cls[0])
                        cls_name = self.model.names[cls_id]
                        confidence = float(box.conf[0])
                        
                        # Only process our target classes with good confidence
                        if cls_name in self.target_classes and confidence > 0.3:
                            # Get bounding box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Draw bounding box on copy of image
                            img_with_box = img.copy()
                            cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img_with_box, f"{cls_name} {confidence:.2f}", 
                                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # Save image with detection
                            filename = f"{self.detection_folder}/{timestamp}_{cls_name}_{confidence:.2f}.jpg"
                            cv2.imwrite(filename, img_with_box)
                            print(f"Saved detection: {cls_name} with confidence {confidence:.2f}")
                            
                            # Check if we need to trigger alarm
                            if cls_name in self.alarm_classes:
                                alarm_triggered = True
                                print(f"Alarm class detected: {cls_name}")
            
            if not detections_found:
                print("No detections in this frame")
                
            return alarm_triggered
            
        except Exception as e:
            print(f"Error in process_frame: {e}")
            return False
    
    def update(self, step_count):
        """Update detection - runs periodically to maintain speed"""
        # Only run detection occasionally to preserve speed
        if step_count % self.detection_interval == 0:
            print(f"Running detection at step {step_count}")
            try:
                alarm_triggered = self.process_frame()
                if alarm_triggered:
                    self.play_alarm()
                return True
            except Exception as e:
                print(f"Detection error: {e}")
        return False