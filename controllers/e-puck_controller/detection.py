from controller import Robot, Camera, Emitter
import numpy as np
import os
import cv2
from datetime import datetime
from ultralytics import YOLO
import threading
import time
import winsound  # Replace playsound with winsound which is more reliable on Windows

class ObjectDetector:
    def __init__(self, robot, communicator=None):
        self.robot = robot
        self.time_step = int(robot.getBasicTimeStep())
        self.communicator = communicator  # Store the robot communicator reference

        # Camera
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

        # Emitter (for backward compatibility)
        try:
            self.emitter = robot.getDevice("emitter")
            if self.emitter:
                print("Emitter initialized")
                self.emitter.setChannel(1)
            else:
                print("Warning: Emitter not found")
        except Exception as e:
            print(f"Emitter not available: {e}")
            self.emitter = None

        # Detections folder
        self.detection_folder = "detections"
        os.makedirs(self.detection_folder, exist_ok=True)

        self.target_classes = ['CardboardBox', 'Cat', 'OilBarrel', 'PlasticCrate', 'WoodenBox']
        self.alarm_classes = ["Cat"]
        self.alarm_sound_file = "mixkit-classic-short-alarm-993.wav"
        self.detection_interval = 100
        self.model = None
        self.alarm_playing = False
        self.alarm_thread = None
        
        # Last detection time (for cooperative detection)
        self.last_detection_times = {cls: 0 for cls in self.target_classes}
        self.robot_position = (0, 0)  # Default position

        print("Object detector initialized with classes:", self.target_classes)
        
    def set_position(self, x, y):
        """Set the current robot position for detection reports"""
        self.robot_position = (x, y)

    def play_robot_alarm(self):
        """Play the alarm sound on the PC using winsound instead of playsound"""
        try:
            print("🔊 Playing alarm on PC...")
            # Check if the sound file exists
            if not os.path.exists(self.alarm_sound_file):
                print(f"⚠️ Sound file not found: {self.alarm_sound_file}")
                return False
                
            # Use winsound.PlaySound which is more reliable than playsound
            winsound.PlaySound(self.alarm_sound_file, winsound.SND_FILENAME)
            return True
        except Exception as e:
            print(f"⚠️ Error playing sound on PC: {e}")
            return False

    def send_alert_emitter(self, message):
        """Send alert message via Emitter (legacy method)"""
        if self.emitter:
            try:
                self.emitter.send(message.encode('utf-8'))
                print(f"📡 Emitter sent message: {message}")
            except Exception as e:
                print(f"⚠️ Failed to send message via emitter: {e}")
        else:
            print("⚠️ No emitter available to send alert")

    def alarm_thread_function(self):
        print("\n" + "!" * 60)
        print("🔊 ALARM! ALARM! Cat detected! 🔊")
        print("!" * 60 + "\n")

        self.play_robot_alarm()
        self.alarm_playing = False

    def play_alarm(self):
        if self.alarm_playing:
            print("⚠️ Alarm already playing, not starting another instance ⚠️")
            return

        self.alarm_playing = True
        self.alarm_thread = threading.Thread(target=self.alarm_thread_function)
        self.alarm_thread.daemon = True
        self.alarm_thread.start()

    def process_frame(self):
        if not self.camera:
            print("No camera available for detection")
            return False, None

        if self.model is None:
            try:
                model_path = "best.pt"
                if not os.path.exists(model_path):
                    print(f"Error: Model file {model_path} not found!")
                    return False, None
                self.model = YOLO(model_path)
                print(f"YOLO model loaded: {self.model.names}")
            except Exception as e:
                print(f"Failed to load YOLO model: {e}")
                return False, None

        try:
            image = self.camera.getImage()
            if not image:
                print("Failed to get image from camera")
                return False, None

            try:
                img = np.frombuffer(image, np.uint8).reshape((self.camera_height, self.camera_width, 4))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            except Exception as e:
                print(f"Error converting image: {e}")
                return False, None

            try:
                results = self.model(img, conf=0.25, verbose=False)
            except Exception as e:
                print(f"YOLO inference error: {e}")
                return False, None

            alarm_triggered = False
            detections_found = False
            detection_results = []

            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    detections_found = True

                    for i, box in enumerate(result.boxes):
                        try:
                            cls_id = int(box.cls[0])
                            cls_name = self.model.names[cls_id]
                            confidence = float(box.conf[0])

                            if cls_name in self.target_classes and confidence > 0.3:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                img_with_box = img.copy()
                                cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(img_with_box, f"{cls_name} {confidence:.2f}",
                                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                filename = f"{self.detection_folder}/{timestamp}_{cls_name}_{confidence:.2f}.jpg"
                                cv2.imwrite(filename, img_with_box)
                                print(f"Saved detection: {cls_name} ({confidence:.2f})")
                                
                                # Add to detection results
                                detection_results.append({
                                    "type": cls_name,
                                    "confidence": confidence,
                                    "box": (x1, y1, x2, y2)
                                })

                                # Update last detection time for this class
                                self.last_detection_times[cls_name] = self.robot.getTime()

                                if cls_name in self.alarm_classes:
                                    alarm_triggered = True
                                    print(f"Alarm class detected: {cls_name}")
                        except Exception as box_err:
                            print(f"Error processing detection box: {box_err}")

            if not detections_found:
                print("No detections in this frame")

            return alarm_triggered, detection_results

        except Exception as e:
            print(f"Error in process_frame: {e}")
            return False, None

    def broadcast_detections(self, detections):
        """Broadcast all detections using the communicator"""
        if not self.communicator or not detections:
            return False
            
        x, y = self.robot_position
        alarm_should_trigger = False
        
        for detection in detections:
            detection_type = detection["type"]
            confidence = detection["confidence"]
            
            # Use the communicator to broadcast the detection
            success = self.communicator.handle_detection_alert(
                detection_type, 
                confidence, 
                x, 
                y
            )
            
            # If it's a cat detection with high confidence, trigger alarm
            if detection_type == "Cat" and confidence > 0.5:
                alarm_should_trigger = True
                
        return alarm_should_trigger  # Return True if we detected a cat

    def update(self, step_count, robot_x=None, robot_y=None):
        if robot_x is not None and robot_y is not None:
            self.set_position(robot_x, robot_y)
            
        if step_count % self.detection_interval == 0:
            print(f"Running detection at step {step_count}")
            try:
                alarm_triggered, detections = self.process_frame()
                
                # If we have detections and a communicator, broadcast them
                if detections and self.communicator:
                    # Broadcast all detections and check if we should alarm
                    should_alarm = self.broadcast_detections(detections)
                    
                    # Only trigger alarm if in cooperative mode we're allowed to
                    if alarm_triggered and should_alarm:
                        self.play_alarm()
                        return True
                elif alarm_triggered:
                    # Legacy mode (no communicator)
                    self.play_alarm()
                    return True
            except Exception as e:
                print(f"Detection error: {e}")
        return False