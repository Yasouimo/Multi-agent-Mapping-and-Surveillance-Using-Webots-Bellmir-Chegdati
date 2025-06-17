from controller import Receiver, Emitter
import json
import os
from datetime import datetime
import numpy as np

class RobotCommunicator:
    """
    Handles low-level communication for the multi-robot team.
    This class is kept simple: it just sends and receives messages.
    The logic for what to do with messages is handled by the controller.
    """
    def __init__(self, robot, channel=1):
        self.robot = robot
        self.robot_name = robot.getName()
        self.time_step = int(robot.getBasicTimeStep())

        # --- Communication Devices ---
        self.emitter = robot.getDevice("emitter")
        self.receiver = robot.getDevice("receiver")
        
        # Emitter setup
        self.emitter.setChannel(channel)
        
        # Receiver setup
        self.receiver.enable(self.time_step)
        self.receiver.setChannel(channel)
        
    def broadcast_message(self, message_payload):
        """
        Broadcasts a message to all other robots on the channel.
        The message is a Python dictionary, which will be converted to a JSON string.
        """
        # Add sender information to the payload
        message_payload["robot_name"] = self.robot_name
        message_payload["timestamp"] = self.robot.getTime()
        
        try:
            # Serialize the dictionary to a JSON string and encode it to UTF-8
            message_string = json.dumps(message_payload)
            self.emitter.send(message_string.encode('utf-8'))
        except Exception as e:
            print(f"[{self.robot_name}] Error sending message: {e}")

    def check_for_messages(self):
        """
        Checks the receiver queue for new messages and parses them from JSON.
        
        Returns:
            A list of dictionaries, where each dictionary is a parsed message.
        """
        messages = []
        while self.receiver.getQueueLength() > 0:
            try:
                # Get the raw data string and parse it from JSON
                data_string = self.receiver.getString()
                message = json.loads(data_string)
                
                # Ignore messages sent by this robot itself
                if message.get("robot_name") != self.robot_name:
                    messages.append(message)
            
            except json.JSONDecodeError:
                # The message received was not valid JSON, ignore it.
                print(f"[{self.robot_name}] Received a malformed message.")
                pass 
            except Exception as e:
                print(f"[{self.robot_name}] Error processing received message: {e}")
            finally:
                # Move to the next message in the queue
                self.receiver.nextPacket()
                
        return messages
