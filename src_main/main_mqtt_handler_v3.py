# Copyright (c) OpenMMLab. All rights reserved.
# Dic 2024
# Source --> https://github.com/mikeligUPM/tfm_edgecloud_registrator/tree/main
#        --> mmsegmentation
# a binary mask is created to get only people and then the pc is created with depth frames
# checkear cuda device y ver que este libre
#! limitar a 1 CPU

import base64
import json
import os
import threading
import numpy as np
import cv2
import open3d as o3d 
from threading import Timer
import paho.mqtt.client as mqtt
from mmseg.apis import inference_model, init_model, show_result_pyplot
from logger_config import logger # logs 
from helper_funs import get_config
from registrator_icp_ransac import icp_p2p_registration_ransac, icp_p2l_registration_ransac
from blob_handler import save_and_upload_pcd

MQTT_BROKER = '138.100.58.224'
MQTT_PORT = 1883
MQTT_TOPIC_CAM = '1cameraframes'
MQTT_QOS = 1
SEND_FRECUENCY = 1
NUM_TOTAL_CAMERAS = 8

received_frames_dict = {}
received_frames_lock = threading.Lock()
batch_timeout = 15 # seconds


def decode_files(color_enc, depth_enc):
    ## RGB file decode  
    decode_c_file_binary = base64.b64decode(color_enc)
    if len(decode_c_file_binary) % 2 != 0:
        decode_c_file_binary += b'\x00'
        logger.debug(f"Adjusted len color_image_data: {len(decode_c_file_binary)}\n")
    frame_color_decoded = cv2.imdecode(np.frombuffer(decode_c_file_binary, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame_color_decoded is None:
        logger.error("Decoded color image is None. Error loaf¡ding image or corverting color image to RGB. Check the data format")
    
    ## Depth file decode  
    decode_d_file_binary = base64.b64decode(depth_enc)
    if len(decode_d_file_binary) % 2 != 0:
        decode_d_file_binary += b'\x00'
        logger.debug(f"Adjusted len depth_image_data: {len(decode_d_file_binary)}\n")
    frame_depth_decoded = cv2.imdecode(np.frombuffer(decode_d_file_binary, dtype=np.uint16), cv2.IMREAD_UNCHANGED)
    if frame_depth_decoded is None:
        logger.error("Decoded depth image is None, check the data format")

    return frame_color_decoded, frame_depth_decoded


# use to create de point cloud from encoded data
def encode_files (new_rgb_name_frame, new_depth_name_frame):
    with open(new_rgb_name_frame, "rb") as color_file:
        encoded_c_string = base64.b64encode(color_file.read()).decode('utf-8')
    with open(new_depth_name_frame, "rb") as depth_file:
        encoded_d_string = base64.b64encode(depth_file.read()).decode('utf-8')
    return encoded_c_string, encoded_d_string


def create_pc_from_enc_data(color_image_enc, depth_image_enc, K, target_ds):
    
    color_image, depth_image = decode_files(color_image_enc, depth_image_enc)
    try:
        # Convert BGR to RGB
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Error converting color image to RGB: {e}")
        
    
    depth_raw = o3d.geometry.Image((depth_image.astype(np.float32) / 1000.0))  # Dividir entre 1000 si está en mm
    color_raw = o3d.geometry.Image(color_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)

    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(color_image.shape[1], color_image.shape[0], K[0][0], K[1][1], K[0][2], K[1][2])

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

    logger.debug(f"PCD len before downsampling {len(pcd.points)}")
    
    #vox_size, _ = get_config(target_ds)
    vox_size = 0.00001
    logger.debug(f"[TEST] Voxel size: {vox_size}")
    pcd = pcd.voxel_down_sample(voxel_size=vox_size)
    
    logger.debug(f"PCD len after downsampling {len(pcd.points)}")

    pcd.estimate_normals()

    return pcd


def apply_mask(out_file_model, rgb_name_frame, depth_name_frame):
    logger.info(f"Creating rgba frame of {os.path.basename(rgb_name_frame)} and depth_a of {os.path.basename(depth_name_frame)}...")
    
    seg_img = cv2.imread(out_file_model)  # img: segmented image
    color_img = cv2.imread(rgb_name_frame)  # img: color image
    depth_img = cv2.imread(depth_name_frame, cv2.IMREAD_UNCHANGED)  # img: depth image, keep original channels

    # Create the mask where people are represented by the color [61, 5, 150]
    mask = np.all(seg_img == [61, 5, 150], axis=-1)  # Boolean mask

    # Create the binary mask image
    binary_mask = np.zeros_like(seg_img)
    binary_mask[mask] = [255, 255, 255]  # White where people are, black elsewhere
    cv2.imwrite(out_file_model, binary_mask)
    logger.info(f"Binary mask of frame {rgb_name_frame} created")

    # Prepare the RGBA color image
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2BGRA)

    # Set the alpha channel: 255 for people, 0 for the background
    color_img[:, :, 3] = mask.astype(np.uint8) * 255 

    # Save the color image with the alpha channel
    cv2.imwrite(rgb_name_frame, color_img)

    # Depth frame with black background 
    depth_img[~mask] = 0
    cv2.imwrite(depth_name_frame, depth_img)

    return rgb_name_frame, depth_name_frame



# Dataset ADE20k: person index = 12 [0-149] 150 classes -> color = [150, 5, 61] 
def segmentator(color_frame_path, camera_name):
    # parameters 
    img_path = color_frame_path # full path from the root 
    config_file = 'configs/pspnet/pspnet_r50-d8_4xb4-80k_ade20k-512x512.py' # model PSPNET, dataset: ADE20k
    checkpoint_file = 'work_dir_el_weno/test_visual/iter_80000.pth' # the model is already trained 
    opacity_value = 1   # set to 1, full opacity to identify colours (classes) better 
    labels_set = False 
    device_selected = 'cuda:7'  # OJO checkear antes 
    tittle = None
    #avg_non_ignore = True

    # build the model from a config file and a checkpoint file
    model = init_model(config_file, checkpoint_file, device = device_selected)

    # save results in folder 
    mask_dict = 'images/mask'
    os.makedirs(mask_dict, exist_ok=True)

    # the result of this process is not the final mask
    result = inference_model(model, img_path) 
    name_mask = img_path.replace('color', 'mask')
    out_file_model = os.path.join(mask_dict, name_mask)

    # SAVE the results
    # return vis_image = show_result_pyplot(...arguments...) --> comprobar
    show_result_pyplot(
        model,
        img_path,
        result,
        title = tittle,
        opacity = opacity_value,
        with_labels = labels_set,
        draw_gt=False,
        show = False ,      # if args.out_file is not None else True,
        out_file = out_file_model,
    )

    logger.info(f"Model inference done. {os.path.basename(color_frame_path)} of CAMERA {camera_name}")
    # segmentator returns the path of the segmented image; result is just for check if there is/isnt smth
    return result, out_file_model 


# process frames of the same number 
def process_frames(msg_frames_list, num_frame):
    ## FIRST: create directories and save all frames of the list msg_frames_list
    base_dict_color = 'images/color'
    os.makedirs(base_dict_color, exist_ok=True)
    base_dict_depth = 'images/depth'
    os.makedirs(base_dict_depth, exist_ok=True)
    i = 1 # controlar 
    pcd_list = [] # deberia contener todas las pc de todas las camaras para un mismo frame, lista de 8 elementos

    for message in msg_frames_list:
        camera_name = message.get('folder_name')
        color_frame_name = message.get('frame_color_name')
        enc_c = message.get('enc_c')
        depth_frame_name = message.get('frame_depth_name')
        enc_d = message.get('enc_d')
        ts = message.get('send_ts')
        container_name = message.get('container_name')
        K = message.get('K') #[[915.76, 0.0, 965.475], [0.0, 916.205, 551.795], [0.0, 0.0, 1.0]] #
        target_ds = 1 # params if were exist more than one method or dataset ///// using now to not modify the rest of scripts 
        #print(K)
    
        frame_color_decoded, frame_depth_decoded = decode_files(enc_c, enc_d) # decode to save images
        
        # save frames
        path_color_frame = os.path.join(base_dict_color, color_frame_name)
        cv2.imwrite(path_color_frame, frame_color_decoded)
        path_depth_frame = os.path.join(base_dict_depth, depth_frame_name)
        cv2.imwrite(path_depth_frame, frame_depth_decoded)
        logger.info(f"Frame (color and depth) saved. Num: {num_frame}. Processing message {i} / {len(msg_frames_list)}")
    
        ## SECOND: preprocessing color frame before creating the pc 
        # segmentation and new rgb image 
        resutl, seg_frame_path = segmentator(path_color_frame, camera_name)
        if resutl is None: 
            logger.error(f"Error in preprocessing frame: {path_color_frame} [SEGMENTATION]. No segmented image")
        else:
            # apply_mask. args: path of the mask and the path of the original rgb image 
            new_rgb_name_frame, new_depth_name_frame = apply_mask(seg_frame_path, path_color_frame, path_depth_frame)
        
        ## PRE-THIRD encode files (reading frames from pahts doesnt work [downsalmpling: ~1million points -> 3 points], create pc from encoding files works)
        enc_c, enc_d = encode_files(new_rgb_name_frame, new_depth_name_frame)
        ## THIRD: create pc from each rgb image
        pc = create_pc_from_enc_data(enc_c, enc_d, K, target_ds)

        save_and_upload_pcd(pc, f"20250131_simple_pc_{camera_name}_{num_frame}.ply", container_name)

        pcd_list.append(pc)
        if pc is None:
            logger.error(f"Error creating point cloud for frame {path_color_frame} of {camera_name}")
            i += 1
            continue
        logger.info(f"[TS] Frame [{color_frame_name}] PCD created for camera {camera_name}")
        i += 1

    print(f"Tamaño de la lista {len(pcd_list)}")

    # delete frames after process the frame 
    #try:
    #    shutil.rmtree('images/')  
    #    logger.info(f"DELETE all images directories")
    #except:
    #    pass

    ## FOURTH: fusion 
    # params if were exist more than one method or dataset ///// using now to not modify the rest of scripts 
    target_registration = 0 # icp_p2p_registration_ransac   # MIKEL: message_json_list[0].get('reg')
    final_fused_point_cloud = icp_p2p_registration_ransac(pcd_list, target_ds) 

    if final_fused_point_cloud is None:
        logger.info(f"Frame [{num_frame}] Final PCD is None. Please check error logs.")
    else:
        logger.debug(f"FRAME [{num_frame}] REGISTRATION SUCCESSFUL")
        
        reg_name = "icp_p2p_ransac" #reg_name = registration_names_from_id.get(target_registration, "unknown")
        dataset_name = "real" #dataset_name_from_id.get(target_ds, "unknown")
        blob_name_reg = f"20250131_final_{num_frame}_{reg_name}_{dataset_name}.ply"
        
        save_and_upload_pcd(final_fused_point_cloud, blob_name_reg, container_name)


# si pasan los x segundos, sigue
def on_batch_timeout(num_frame):
    with received_frames_lock:
        logger.info(f"Timeout for frame {num_frame} detected")
        if num_frame in received_frames_dict and received_frames_dict[num_frame][0]:
            received_frames_dict[num_frame][1].cancel()  # Stop the timer
            frame_data_copy = received_frames_dict.pop(num_frame)[0]
            threading.Thread(target=process_frames, args=(frame_data_copy, num_frame)).start()


# process_message function extracts all information from the message received 
def process_message(msg):
    # get payload only camera name and frame 
    message = json.loads(msg.payload)
    camera_name = message.get('folder_name')
    color_frame_name = message.get('frame_color_name')
    container_name = message.get('container_name')
    _,_,num_frame = (color_frame_name.split('.')[0]).split('_') # name frame: cameraname_typeframe_numberframe.png
    logger.info(f"[TS] Received MSG with frame {num_frame} of the camera {camera_name} and sequence {container_name}")
    
    with received_frames_lock:
        if num_frame not in received_frames_dict: # llega un NUEVO frame, inicio un temporizador 
            received_frames_dict[num_frame] = ([], Timer(batch_timeout, on_batch_timeout, args=(num_frame)))
            received_frames_dict[num_frame][1].start() # set the timer 
        
        received_frames_dict[num_frame][0].append(message) # guardo todos los frames que llegan, num_frame es la CLAVE

        if len(received_frames_dict[num_frame][0]) == NUM_TOTAL_CAMERAS: 
            logger.info(f"Batch full for frame {num_frame}")
            received_frames_dict[num_frame][1].cancel()  # Stop the timer
            frame_data_copy = received_frames_dict.pop(num_frame)[0]
            threading.Thread(target=process_frames, args=(frame_data_copy, num_frame)).start()
        else:
            # As a new event has arrived, reset timer 
            if received_frames_dict[num_frame][1] is not None:
                received_frames_dict[num_frame][1].cancel()
            received_frames_dict[num_frame] = (received_frames_dict[num_frame][0], Timer(batch_timeout, on_batch_timeout, args=[num_frame]))
            received_frames_dict[num_frame][1].start()


## MQTT funs 
def on_connect(client, userdata, flags, rc):
    logger.info(f"Connected with result code {rc}")
    client.subscribe(MQTT_TOPIC_CAM)

def on_message(client, userdata, msg): 
    threading.Thread(target=process_message, args=(msg,)).start()

def main():
    # Connection to MQTT broker
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(MQTT_BROKER, MQTT_PORT, 60)     
    client.loop_forever()

main()
