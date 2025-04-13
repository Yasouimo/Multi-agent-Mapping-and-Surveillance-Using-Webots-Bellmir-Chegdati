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
        
        # Get speaker
        self.speaker = robot.getDevice("speaker") if hasattr(robot, 'getDevice') else None
        
        # Create detection folder
        self.detection_folder = "detections"
        os.makedirs(self.detection_folder, exist_ok=True)
        
        # Classes we're interested in
        self.target_classes = ["rock", "dog", "cat", "oil_barrel", "plastic_crate", "wooden_box", "cardboard_box"]
        
        # Alarm-triggering classes
        self.alarm_classes = ["cat", "dog"]
        
        # Alarm sound file
        self.alarm_sound = "mixkit-classic-short-alarm-993.wav"
        
        # Detection frequency - run rarely to maintain speed
        self.detection_interval = 100  # Run detection every 100 steps
        self.last_detection = 0
        
        # Load model lazily
        self.model = None
    
    def play_alarm(self):
        """Play alarm for cat or dog detection"""
        try:
            if self.speaker:
                # Correct Webots API call with all required arguments
                self.speaker.playSound(self.speaker, self.alarm_sound, 1.0, 1.0, 0, 0)
                print("⚠️ ALARM: Cat or dog detected! ⚠️")
            else:
                print("⚠️ ALARM: Cat or dog detected! (No speaker available) ⚠️")
        except Exception as e:
            print(f"⚠️ ALARM: Cat or dog detected! (Speaker error: {e}) ⚠️")
    
    def process_frame(self):
        """Process camera frame and save detected objects as images"""
        # Skip if no camera
        if not self.camera:
            return False
        
        # Load model if needed
        if self.model is None:
            try:
                self.model = YOLO("best.pt")
            except Exception as e:
                print(f"Failed to load YOLOv8 model: {e}")
                return False
        
        # Get image from camera
        image = self.camera.getImage()
        if not image:
            return False
        
        # Convert to OpenCV format
        img = np.frombuffer(image, np.uint8).reshape((self.camera_height, self.camera_width, 4))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Run detection (silently)
        results = self.model(img, conf=0.25, verbose=False)
        
        # Check for detections
        alarm_triggered = False
        if results and len(results) > 0:
            # Get result data
            result = results[0]
            if len(result.boxes) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Process each detection
                for i, box in enumerate(result.boxes):
                    # Get class info
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    confidence = float(box.conf[0])
                    
                    # Only process our target classes
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
        
        return alarm_triggered
    
    def update(self, step_count):
        """Update detection - runs periodically to maintain speed"""
        # Only run detection occasionally to preserve speed
        if step_count % self.detection_interval == 0:
            try:
                alarm_triggered = self.process_frame()
                if alarm_triggered:
                    self.play_alarm()
                return True
            except Exception as e:
                print(f"Detection error: {e}")
        return False