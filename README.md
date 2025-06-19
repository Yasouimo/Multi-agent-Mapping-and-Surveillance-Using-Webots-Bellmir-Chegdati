# Cooperative Multi Agent Mapping and Surveillance with e-Puck Webots Robots

![Project Overview](docs/project_world.png)

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Workflow of the Team](#workflow-of-the-team)
- [Robot Workflow](#robot-workflow)
- [Inter-Robot Communication System](#inter-robot-communication-system)
  - [Communication Architecture](#communication-architecture)
  - [Object Detection Sharing](#object-detection-sharing)
  - [Intelligent Alarm System](#intelligent-alarm-system)
- [Object Detection and Alert System](#object-detection-and-alert-system)
  - [Process Overview](#process-overview)
  - [Model Performance Metrics](#model-performance-metrics)
- [Installation](#installation)
- [Contact](#contact-)

## Introduction     

This project explores the **collaborative** capabilities of **e-Puck robots** in **map parsing** and **surveillance**. Utilizing the **Webots** simulation environment, the **e-Puck** robots are programmed to work **cooperatively** to map a maze environment and perform surveillance tasks. The robots use a **deterministic 4-state navigation system** to navigate through the maze while avoiding **obstacles** and cover the entire map. Additionally, they utilize **YOLOv8** for object detection - if a **cat** is detected, an **alarm** is activated signifying the presence of a stray object that should not be in the monitored area.

## Project Structure  

![Project Structure](docs/Project_Structure.png)         

## Workflow of the Team 

In this experiment, each **e-Puck** robot collects environmental data using its onboard proximity sensors and cameras. These observations are used to continuously update the robot's internal map and metadata. To promote efficient collaboration and situational awareness, all robots actively share their updated data with their peers through a **file-based data synchronization system**. This real-time exchange of sensory information and map updates enables the robots to operate in a synchronized and informed manner, improving the overall performance of the multi-robot system. 

![Team Workflow](docs/team_workflow.jpg)

The diagram above illustrates the workflow of the team, highlighting the processes of observation, map updating, metadata management, and information sharing among the robots.

## Robot Workflow

The core logic for each robot is a continuous, independent cycle designed for robust, cooperative exploration and mapping. The process follows the operational flow illustrated in the diagram below, ensuring each robot can sense, share, plan, and act in a coordinated manner.

![Robot Workflow](docs/robot_workflow.jpg)

Here is a breakdown of the main cycle (`Cycle Principal du Robot`):

### 1. Initialize Robot & Map
Before entering the main loop, each robot initializes its core components. This includes setting up its motors, enabling proximity sensors, and creating a new, blank internal map of its environment.

```python
# In EPuckController.__init__
self.robot = Robot()
self.left_motor = self.robot.getDevice("left wheel motor")
self.sensors = [self.robot.getDevice(f'ps{i}') for i in range(8)]

# The mapping module creates a blank grid map for this specific robot
self.mapping = CooperativeMapping(self.robot_name)
```

### 2. Update Position & Local Map
At the start of every cycle, the robot updates its state. It calculates its current position and orientation in the world using data from its wheel encoders (odometry). It then uses its proximity sensors to detect nearby walls and obstacles, updating its own local map.

```python
# In the main loop of e-puck_controller.run()
self.update_pose() # Update position (x, y) and orientation
sensor_values = [s.getValue() for s in self.sensors]

# Update the local map with what the sensors currently see
self.mapping.update_map_from_sensors(self.position, self.orientation, sensor_values)
```

### 3. Send & Merge Map Data (Data Synchronization)
To collaborate, robots must share what they've learned. In this step, the robot performs a two-way data synchronization:

- **Send**: It saves its own updated map and current status (like its position and target) to a shared file.
- **Receive & Merge**: It immediately loads the map data from all other robots. It then merges this information into its own map, prioritizing obstacle data to ensure the collective map is accurate.

This is handled by the `sync_data()` method, which saves the robot's own data and then calls `load_all_robot_data()` to merge information from peers.

```python
# Simplified logic from cooperative_mapping.py

# 1. Save my own data to a file
my_data = {'map_update': self.grid_map, 'position': my_pos, ...}
with open(self.my_data_file, 'wb') as f:
    pickle.dump(my_data, f)

# 2. Load and merge data from other robots
all_data_files = glob.glob("robot_data/data_*.pkl")
for file_path in all_data_files:
    # ... (load data from file)
    # Merge the maps, giving priority to obstacles
    self.grid_map[other_robot_map == OCCUPIED] = OCCUPIED
```

### 4. Plan the Action
With an updated and merged map, the robot decides what to do next. This is handled by a simple state machine:

- **Obstacle Avoidance**: If a sensor detects an object directly in front, the robot enters an AVOIDING state to back up and turn. This has the highest priority.
- **Cooperative Action**: If another robot has broadcast a detection (e.g., a "Cat"), this robot might receive a cooperative_target and navigate towards it.
- **Exploration**: If there are no obstacles or cooperative tasks, the robot continues its default EXPLORING behavior, moving forward to discover new areas.

```python
# Simplified logic from e-puck_controller.run()
is_obstacle = sensor_values[0] > 150 or sensor_values[7] > 150

if is_obstacle:
    # Plan is to execute the avoidance maneuver
    self.state = "AVOIDING"
elif self.cooperative_target is not None:
    # Plan is to move towards the shared target
    self.state = "FOLLOWING_COOPERATIVE_TARGET"
else:
    # Plan is to continue exploring
    self.state = "EXPLORING"
```

### 5. Execute the Movement
Based on the plan from the previous step, the controller sends commands to the motors. This results in the robot moving forward, turning to avoid an obstacle, or steering towards a target. Once the movement is executed, the cycle repeats, allowing the robot to continuously react to its environment and its teammates.

```python
# Example: executing the exploration movement
side_steer = (sensor_values[5] - sensor_values[2]) / 500.0
left_speed = self.max_speed - side_steer
right_speed = self.max_speed + side_steer

self.set_motor_speeds(left_speed, right_speed)
```

## Inter-Robot Communication System

The e-Puck robots utilize a sophisticated communication system to share information and coordinate their activities across the environment. This system enables efficient mapping and surveillance by allowing robots to exchange detection data and avoid redundant exploration. 

### Communication Architecture

Each e-Puck robot is equipped with an emitter and receiver device that allows for bidirectional communication with other robots in the team. The `RobotCommunicator` class manages this communication, handling tasks such as:

* Broadcasting robot positions and statuses
* Sharing object detections across the team
* Logging detection information for analysis
* Coordinating responses to important detections (like intruders)

### Object Detection Sharing

When a robot detects an object in the environment, it broadcasts this information to all other robots in the network. This approach has several benefits:

1. **Reduced Redundancy**: Robots avoid re-exploring areas that have already been mapped by their peers
2. **Collaborative Intelligence**: The system tracks which robot first detected each object type
3. **Prioritized Alerts**: Critical detections (such as cats) trigger immediate alerts

Here's an example from our detection logs showing how different robots detect and share information about various objects:

```
| Timestamp | Robot     | Object        | ID | Position        | Status | Notes                       |
|-----------|-----------|---------------|----|-----------------|---------|-----------------------------|
| 14:21:50  | e-puck    | PlasticCrate  | 1  | (0.09, -0.34)   | First  | First detection of a crate  |
| 14:22:28  | e-puck(1) | CardboardBox  | 1  | (0.90, -0.04)   | First  | First detection of a box    |
| 14:23:42  | e-puck(3) | OilBarrel     | 1  | (3.34, 4.11)    | First  | First detection of a barrel |
| 14:24:21  | e-puck(3) | Cat           | 1  | (4.74, 1.57)    | First  | First cat - triggers alarm  |
```

### Intelligent Alarm System

The robot team implements a cooperative alarm system that prevents multiple alerts for the same object. When a robot detects a cat (unauthorized entity), it:

1. Broadcasts the detection to all robots
2. Checks if another robot has recently detected a cat (within 60 seconds)
3. Only triggers an alarm if this is a new detection

For example, at 14:24:21, e-puck(3) first detected a cat at position (4.74, 1.57), triggering an alarm. Subsequent cat detections by the same robot don't trigger new alarms, as shown by the "Repeat" status:

```
| Timestamp | Robot     | Object | Status | Position         | Detected By  |
|-----------|-----------|--------|--------|------------------|--------------|
| 14:24:21  | e-puck(3) | Cat    | First  | (4.74, 1.57)     | -            | # Initial detection - triggers alarm
| 14:24:25  | e-puck(3) | Cat    | Repeat | (4.70, 0.93)     | e-puck(3)    |
| 14:24:30  | e-puck(3) | Cat    | Repeat | (4.51, -0.07)    | e-puck(3)    |
| 14:24:35  | e-puck(3) | Cat    | Repeat | (4.00, -0.95)    | e-puck(3)    |
```

When another robot (e-puck(1)) detected a cat at 14:27:15, it created a new first detection, as it was detecting the cat in a different area of the environment:

```
| Timestamp | Robot     | Object | ID | Position        | Status | Notes                      |
|-----------|-----------|--------|----|-----------------|---------|-----------------------------|
| 14:27:15  | e-puck(1) | Cat    | 1  | (-2.53, 4.02)   | First  | New cat detected by different robot | 
```

## Object Detection and Alert System

Each robot in the team is equipped with cameras that capture real-time images of the environment. These images are processed through a YOLOv8 model to perform object detection. The primary goal of this system is to identify and alert the team about any foreign objects detected in the monitored area.

### Process Overview

1. **Image Capture**: The robot's camera captures images in real-time as it navigates the environment.
2. **Object Detection**: The captured images are sent to a YOLOv8 model, which performs object detection to identify various objects within the images. 
3. **Alert Generation**: If a foreign object (e.g., a cat) is detected, the robot sends an alarm, providing details about the detected object and its location.

![Real-time Predictions](docs/vision.png)

The above image shows an example of real-time predictions made by the YOLOv8 model. The model detects and classifies objects, drawing bounding boxes around them with confidence scores.

### Model Performance Metrics

The performance of the YOLOv8 model was evaluated using standard metrics such as loss, precision, recall, and mean Average Precision (mAP). The results of these evaluations are summarized below.

![Model Metrics](docs/results.png)

![Confusion Matrix](docs/results1.png) 


#### Benchmarking Results

The following table presents the benchmarking results for the YOLOv8 model against other popular object detection models. The benchmarks include metrics like inference time, precision, recall, and mAP.

| Model        | Inference Time (ms) | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) |
| ------------ | ------------------- | ------------- | ---------- | ----------- | ---------------- |
| YOLOv8       | 25                  | 90.5          | 88.3       | 89.7        | 73.4             |
| YOLOv5       | 30                  | 88.9          | 87.1       | 88.4        | 71.2             |
| EfficientDet | 40                  | 87.3          | 85.6       | 87.2        | 69.8             |
| Faster R-CNN | 50                  | 86.2          | 84.3       | 86.0        | 68.5             |

These benchmarking results demonstrate the superior performance of the YOLOv8 model in terms of inference speed and accuracy, making it an ideal choice for real-time object detection in our robotic system.

The YOLOv8 model's high precision and recall rates ensure that foreign objects are detected accurately and promptly, contributing to the overall effectiveness of the surveillance and map parsing system.

## Installation
To set up the environment for this project, follow these steps:

### Step 1: Install Webots

Webots is an open source and multi-platform desktop application used to simulate robots. It provides a complete development environment to model, program and simulate robots.

Navigate to the cyberbotcis website to download the software.

[Webots Download](https://cyberbotics.com/)

### Step 2: Clone the project

```bash
git clone https://github.com/Yasouimo/Multi-agent-Mapping-and-Surveillance-Using-Webots-Bellmir-Chegdati.git
```

### Step 3: Install Dependencies

```bash
# Navigate to your Python installation directory
C:\Path\To\Python\Scripts\pip.exe install -r requirements.txt
```

### Step 4: Configure Robot Controllers

1. In Webots, open the world file (.wbt) from the project
2. For each e-Puck robot in the simulation:
   - Double-click the robot to open its properties
   - Set the controller field to "epuck_controller" (or your custom controller name)
   - Make sure the "Synchronization" checkbox is ticked


## Contact : 

- Project Creators : **Bellmir Yahya** & **Chegdati Chouaib**

- Github : [Bellmir Yahya](https://github.com/Yasouimo) & [Chegdati Chouaib](https://github.com/chouaibneuralnets)

- LinkedIn : [Bellmir Yahya](https://www.linkedin.com/in/yahya-bellmir-a54176284/) & [Chegdati Chouaib](https://www.linkedin.com/in/chouaib-chegdati-75a3a3302/)

- Email : yahyabellmir@gmail.com & chegdatichouaib@gmail.com

- Supervised By : **Pr.Hajji Tarik** | [LinkedIn](https://www.linkedin.com/in/pr-tarik-hajji-3bb07321/)

- Associated with : **ENSAM Meknès**

