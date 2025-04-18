from controller import Robot, Camera, Speaker
import numpy as np
import os
import cv2
from datetime import datetime
from ultralytics import YOLO
import subprocess
import platform
import threading
import time

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
        
        # Speaker initialization (keeping for compatibility)
        try:
            self.speaker = robot.getDevice("speaker")
            if self.speaker:
                print("Robot speaker initialized")
                self.speaker.enable(self.time_step)
        except Exception as e:
            print(f"Robot speaker not available: {e}")
            self.speaker = None
        
        # Create detection folder
        self.detection_folder = "detections"
        os.makedirs(self.detection_folder, exist_ok=True)
        
        # Updated classes to match your YOLO model's class names
        self.target_classes = ['CardboardBox', 'Cat', 'OilBarrel', 'PlasticCrate', 'WoodenBox']
        
        # Alarm-triggering classes - only Cat from your class list
        self.alarm_classes = ["Cat"]
        
        # Updated alarm sound file path (ensure this file exists in the project directory)
        self.alarm_sound_file = "mixkit-classic-short-alarm-993.wav"
        
        # Detection frequency
        self.detection_interval = 100  # Run detection every 100 steps
        
        # Load model lazily
        self.model = None
        
        # Track if alarm is currently playing to avoid overlap
        self.alarm_playing = False
        self.alarm_thread = None
        
        # Operating system check for PC sound playback method
        self.system = platform.system()
        
        print("Object detector initialized with classes:", self.target_classes)
    
    def play_pc_sound(self):
        """Play alarm sound through PC speakers"""
        if not os.path.exists(self.alarm_sound_file):
            print(f"⚠️ Warning: Alarm sound file {self.alarm_sound_file} not found! ⚠️")
            return False
            
        try:
            if self.system == "Windows":
                # Simple method - just use the system default program to play the sound
                # Get the absolute path and normalize it
                sound_path = os.path.abspath(self.alarm_sound_file).replace('\\', '\\\\')
                
                # Use a simpler method in PowerShell to play the sound
                ps_command = f'Start-Process -FilePath "{sound_path}" -WindowStyle Hidden'
                subprocess.call(['powershell', '-Command', ps_command])
                
                # Alternative fallback if the above fails - use the built-in Windows beep
                try:
                    import winsound
                    winsound.Beep(1000, 1000)  # 1000 Hz for 1 second
                except:
                    pass
                    
            elif self.system == "Darwin":  # macOS
                # macOS - use afplay
                subprocess.call(['afplay', self.alarm_sound_file])
            else:  # Linux and others
                # Try with various players available on Linux
                for player in ['mpg123', 'mplayer', 'ffplay', 'paplay', 'aplay']:
                    try:
                        if player == 'ffplay':
                            # ffplay has different syntax
                            subprocess.call([player, '-nodisp', '-autoexit', self.alarm_sound_file])
                        else:
                            subprocess.call([player, self.alarm_sound_file])
                        return True
                    except FileNotFoundError:
                        continue
            return True
            
        except Exception as e:
            print(f"⚠️ Error playing PC sound: {e} ⚠️")
            # ASCII Bell character - should make a beep on most terminals
            print('\a')
            return False
    
    def alarm_thread_function(self):
        """Function to run in a separate thread for playing the alarm"""
        print("\n" + "!" * 60)
        print("🔊 ALARM! ALARM! Cat detected! 🔊")
        print("!" * 60 + "\n")
        
        result = self.play_pc_sound()
        time.sleep(2)  # Wait a moment before allowing another alarm
        self.alarm_playing = False
    
    def play_alarm(self):
        """Play alarm for cat detection"""
        if self.alarm_playing:
            print("⚠️ Alarm already playing, not starting another instance ⚠️")
            return
            
        self.alarm_playing = True
        
        # Start a thread to play the alarm sound
        self.alarm_thread = threading.Thread(target=self.alarm_thread_function)
        self.alarm_thread.daemon = True
        self.alarm_thread.start()
    
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
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    detections_found = True
                    
                    # Process each detection
                    for i, box in enumerate(result.boxes):
                        try:
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
                        except Exception as box_err:
                            print(f"Error processing detection box: {box_err}")
            
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