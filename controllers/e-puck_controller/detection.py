from controller import Robot, Camera
import numpy as np
import os
import cv2
from datetime import datetime
from ultralytics import YOLO
import threading
import winsound
import time

class ObjectDetector:
    def __init__(self, robot, communicator=None):
        self.robot = robot
        self.time_step = int(robot.getBasicTimeStep())
        self.communicator = communicator

        # Camera setup
        self.camera = robot.getDevice("camera")
        if self.camera:
            self.camera.enable(self.time_step)
            self.camera_width = self.camera.getWidth()
            self.camera_height = self.camera.getHeight()
            print(f"Camera initialized: {self.camera_width}x{self.camera_height}")
        else:
            print("Warning: Camera not found")
            self.camera = None

        # Detection settings
        self.detection_folder = "detections"
        os.makedirs(self.detection_folder, exist_ok=True)
        self.target_classes = ['CardboardBox', 'Cat', 'OilBarrel', 'PlasticCrate', 'WoodenBox']
        self.alarm_classes = ["Cat"]
        self.alarm_sound_file = "mixkit-classic-short-alarm-993.wav"
        self.detection_interval = 100  # Run detection every 100 steps
        self.model = None
        self.robot_position = (0, 0)
        
        # Alarm state
        self.alarm_playing = False
        self.alarm_thread = None

    def load_model(self):
        """Loads the YOLO model."""
        if self.model is None:
            try:
                model_path = "best.pt"
                if not os.path.exists(model_path):
                    print(f"Error: Model file '{model_path}' not found!")
                    return False
                self.model = YOLO(model_path)
                print(f"YOLO model loaded with classes: {self.model.names}")
                return True
            except Exception as e:
                print(f"Failed to load YOLO model: {e}")
                return False
        return True

    def play_alarm_sound(self):
        """Plays the alarm sound in a separate thread to avoid blocking."""
        def alarm_task():
            print("\n" + "!" * 60)
            print("      🚨 🐱  ALARM! ALARM! A CAT HAS BEEN DETECTED! 🐱 🚨")
            print("!" * 60 + "\n")
            try:
                if os.path.exists(self.alarm_sound_file):
                    # Play sound asynchronously to not halt the robot
                    winsound.PlaySound(self.alarm_sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
                    time.sleep(3) # Cooldown to prevent sound spam
                else:
                    print(f"Warning: Alarm sound file not found at '{self.alarm_sound_file}'")
            except Exception as e:
                print(f"Error playing alarm sound: {e}")
            finally:
                self.alarm_playing = False

        if not self.alarm_playing:
            self.alarm_playing = True
            self.alarm_thread = threading.Thread(target=alarm_task)
            self.alarm_thread.daemon = True
            self.alarm_thread.start()

    def process_frame(self):
        """Captures an image, runs detection, saves results, and broadcasts."""
        if not self.camera or not self.load_model():
            return False, []

        image_data = self.camera.getImage()
        if not image_data:
            return False, []

        img = np.frombuffer(image_data, np.uint8).reshape((self.camera_height, self.camera_width, 4))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        results = self.model(img_bgr, conf=0.3, verbose=False)
        
        alarm_triggered = False
        detections_made = []
        
        if results:
            result = results[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                confidence = float(box.conf[0])

                if cls_name in self.target_classes:
                    print(f"  -> Detected {cls_name} with confidence {confidence:.2f}")

                    # --- FIXED: Draw bounding box and save the image ---
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Create a fresh copy to draw on for each detection
                    img_with_box = img_bgr.copy()
                    cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{cls_name} {confidence:.2f}"
                    cv2.putText(img_with_box, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Save the image with a unique filename
                    filename = f"{self.detection_folder}/{timestamp}_{i}_{cls_name}.jpg"
                    cv2.imwrite(filename, img_with_box)
                    print(f"   -> Saved detection image to {filename}")

                    # Append to list of detections this frame
                    detections_made.append({"class": cls_name, "confidence": confidence})

                    # Broadcast every detection
                    if self.communicator:
                        self.communicator.broadcast_detection(cls_name, confidence, self.robot_position)
                    
                    if cls_name in self.alarm_classes:
                        alarm_triggered = True

        return alarm_triggered, detections_made

    def update(self, step_count, robot_x, robot_y):
        """Main update loop called by the controller."""
        self.robot_position = (robot_x, robot_y)

        if step_count % self.detection_interval == 0:
            print(f"\n--- Running Object Detection (Step {step_count}) ---")
            try:
                alarm_should_sound, detections = self.process_frame()
                if alarm_should_sound:
                    self.play_alarm_sound()
            except Exception as e:
                print(f"Error during detection cycle: {e}")
