# https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.youtube.com/watch%3Fv%3DSFuaNcyWV8g&ved=2ahUKEwjcsbS9h7aKAxU3VaQEHdZMKFAQtwJ6BAgKEAI&usg=AOvVaw3_JyZtw05QO0f18z7f2zjY
# Jan 2025
# Source --> https://github.com/mikeligUPM/tfm_edgecloud_registrator/tree/main
# CAMERAS SIMULATION. This script sends all frames of 1 camera to the broker under MQTT - TOPIC 'cameraframes'
# OJO: las imagenes capturadas tienen que estar identificadas por el nombre, su nombre tiene que tener color o depth y numero de frame
# --data
#      |--cameras
#              |--000442922112    
#                          |--color
#                          |--depth
# There is an external script "rename_files.py" that given a full path to the files rename all of them 

import os
import base64
import json
from argparse import ArgumentParser
import time
import threading
from logger_config import logger
import paho.mqtt.client as mqtt
from datetime import datetime, timezone
import numpy as np
import open3d as o3d


MQTT_BROKER = "138.100.58.224" # broker en el pc del gdem
MQTT_PORT = 1883  # Todo: Add security to MQTT port 8883
MQTT_TOPIC_CAM = '1cameraframes' 
MQTT_QOS = 1 
SEND_FREQUENCY = 1  # Time in seconds between sending messages

logger.info(f"Camera simulation started with:\nBROKER_IP: {MQTT_BROKER}\nBROKER_PORT: {MQTT_PORT}\nSEND_FREQUENCY: {SEND_FREQUENCY}")

K = [
    [585.0, 0.0, 320.0],
    [0.0, 585.0, 240.0],
    [0.0, 0.0, 1.0]
]

# camera parameters
def create_k_dict_by_camera(filepath) -> dict:
    k_dict = {}
    K = np.eye(3)
    with open(filepath, "r") as f:
        data = json.load(f)
        for _, camera in enumerate(data["cameras"]):
            # Extract camera parameters
            resolution = camera["Resolution"]
            focal = camera["Focal"]
            principal_point = camera["Principle_point"]
            camera_name = camera["Name"]

            # Create PinholeCameraIntrinsic object
            K = o3d.camera.PinholeCameraIntrinsic(
                width=resolution[0],
                height=resolution[1],
                fx=focal[0],
                fy=focal[1],
                cx=principal_point[0],
                cy=principal_point[1]
            )
            k_dict[camera_name] = K.intrinsic_matrix.tolist()
    return k_dict

def save_json_to_local(payload, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(payload, json_file, indent=4)

# Function to encode and transmit files via MQTT msg
def encode_png_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# Calculate size of JSON-encoded payload in bytes
def get_message_size(payload):
    return len(json.dumps(payload))

# Function to construct and send message to IoT Hub 
## message: color y profundidad
# folder_name --> color/depth
# camera_name --> la unica que hay 
def build_publish_encoded_msg(client, camera_name, k, color_name, encoded_color_file, depth_name, encoded_depth_file, dataset_id):
    dt_now = datetime.now(tz=timezone.utc) 
    send_ts = round(dt_now.timestamp() * 1000)

    payload = {
        "folder_name": camera_name,
        "frame_color_name": color_name,
        "enc_c": encoded_color_file,
        "frame_depth_name": depth_name,
        "enc_d": encoded_depth_file,
        "K": k,
        "reg": 0,
        "ds": dataset_id,
        "send_ts": send_ts # UTC timestamp
    }

    # Calculate message size
    message_size = get_message_size(payload)
    logger.info(f"Message size: {message_size} bytes")

    client.publish(MQTT_TOPIC_CAM, json.dumps(payload), qos=MQTT_QOS) # qos=1 --> PUBLISH; PULBACK. if not, send again
    logger.info(f"[TS] Camera [{camera_name}] sent message to IoT Hub, color: {color_name}, depth {depth_name}")


def process_frames_of_a_camera(client, k_dict, camera_name_path, dataset_id): 
    camera_name = os.path.basename(camera_name_path) # 0004422112
    logger.info(f"Sending all frames of camera {camera_name} INIT")

    directories = [os.path.join(camera_name_path, d) for d in os.listdir(camera_name_path) if os.path.isdir(os.path.join(camera_name_path, d))]

    for dir in directories:
        if 'color' in os.path.basename(dir):
            path_color = dir
            color_frames = sorted(f for f in os.listdir(dir) if f.startswith(f"{camera_name}_color_"))
        if 'depth' in os.path.basename(dir):
            path_depth = dir
            depth_frames = sorted(f for f in os.listdir(dir) if f.startswith(f"{camera_name}_depth_"))

    if isinstance(k_dict, dict): # if k_dict es un dicc o lista 
        k_list = k_dict[camera_name]
    elif isinstance(k_dict, list):
        k_list = k_dict

    for chosen_color_frame, chosen_depth_frame in zip(sorted(color_frames), sorted(depth_frames)):
        encoded_color_file = encode_png_to_base64(os.path.join(path_color, chosen_color_frame))
        encoded_depth_file = encode_png_to_base64(os.path.join(path_depth, chosen_depth_frame))
        build_publish_encoded_msg(client, camera_name, k_list, chosen_color_frame, encoded_color_file, chosen_depth_frame, encoded_depth_file, dataset_id)
    
    logger.info(f"[Sending all frames of camera {camera_name} END")
    
# Function to control the flow and send frames and files 
# nota: base_directory = data/cameras
def start_cam_simulation(client, base_directory, dataset_id, send_freq = 3):
    exit_sim = False # ESC
    filepath = 'src_edgeDevice/cam_params.json'
    k_dict = create_k_dict_by_camera(filepath)
    try:
        while not exit_sim:
            camera_name_directories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
            # data/cameras/000442922112 lo unico que hay 
            threads = []  
            for cam_name_dir in camera_name_directories: 
                thread = threading.Thread(target=process_frames_of_a_camera, args=(client, k_dict, cam_name_dir, dataset_id))
                threads.append(thread)
                thread.start()
                time.sleep(0.1) 
                    
            for thread in threads: # wait threads
                thread.join()
            
            time.sleep(send_freq)  # Esperar N segundos antes de comenzar nuevamente
            x = input("continue? ") 

    except KeyboardInterrupt:
        exit_sim = True

# MQTT Publish function 
def on_publish(client, userdata, mid):
    logger.info(f"Message published successfully with MID: {mid}")

# MAIN 
if __name__ == "__main__":
    try:
        # Connection to MQTT broker
        client = mqtt.Client()
        client.on_publish = on_publish
        client.connect(MQTT_BROKER, MQTT_PORT)
        time.sleep(4) # wait for connection setup to complete 
        client.loop_start()
    except Exception as e:
        logger.error(f"Could not connect to broker: {e}")
    else:
        # Starting data publication
        logger.info("Connected.")
        base_directory = 'data/cameras/'
        dataset_id = 1 # no quitar (point clouds)
        x = input("Press ENTER ") 
        start_cam_simulation(client, base_directory, dataset_id, send_freq=SEND_FREQUENCY) # send frames
        logger.info("Simulation ended")