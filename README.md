# Multi Agent Mapping and Surveillance with e-Puck Webots Robots

![Project Overview](docs/project_world.png)


## Introduction

This project explores the collaborative capabilities of e-Puck robots in map parsing and surveillance. Utilizing the Webots simulation environment, the e-Puck robots are programmed to work cooperatively to map a maze environment and perform surveillance tasks. The robots use Q-learning to navigate through the maze while avoiding obstacles and cover the entire map. Additionally, they utilize YOLOv8 for object detection - if a cat is detected, an alarm is activated signifying the presence of a stray object that should not be in the monitored area.

## Workflow of the Team

In this experiment, each e-Puck robot reads observations through proximity sensors and cameras. The robots continuously update their maps and metadata based on these observations. To ensure effective and coordinated operation, all robots share this information with their peer robots.

![Team Workflow](docs/team_workflow.jpg)

The diagram above illustrates the workflow of the team, highlighting the processes of observation, map updating, metadata management, and information sharing among the robots.

## Robot Workflow

Each robot in the team performs a series of steps to ensure effective map parsing and surveillance. The primary tasks performed by each robot are as follows:

1. **Send Map Updates**: Robots calculate updates based on their sensor data and send these updates to their peers.
2. **Receive Map Updates**: Robots receive updates from their peers and integrate this information into their own maps.
3. **Path Planning**: Robots plan their paths based on the updated maps and the positions of their peers to avoid obstacles and cover the area efficiently.
4. **Path Execution**: Robots execute the planned paths, adjusting their movements based on real-time sensor data.

![Robot Workflow](docs/robot_workflow.jpg)

The diagram above illustrates the detailed workflow of the robots, showing the processes of sending and receiving map updates, path planning, and path execution.

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

## Installation

### Step 1: Clone the project

```bash
git clone https://github.com/Yasouimo/Multi-agent-Mapping-and-Surveillance-Using-Webots-Bellmir-Chegdati.git
```

### Step 2: Install Dependencies

```bash
# Navigate to your Python installation directory
C:\Path\To\Python\Scripts\pip.exe install -r requirements.txt
```

### Step 3: Configure Robot Controllers

1. In Webots, open the world file (.wbt) from the project
2. For each e-Puck robot in the simulation:
   - Double-click the robot to open its properties
   - Set the controller field to "epuck_controller" (or your custom controller name)
   - Make sure the "Synchronization" checkbox is ticked
