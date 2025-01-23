# Copyright (c) OpenMMLab. All rights reserved.
# Jan 2025
# Source --> https://github.com/mikeligUPM/tfm_edgecloud_registrator/tree/main
#        --> mmsegmentation
# version nueva para no joder el otro 
# checkear cuda device y ver que este libre
# ver como no guardar las images, recibir ya diezmadas -> pre/procesar -> y fuera  

import base64
import datetime
import json
import os
import threading
import numpy as np
import matplotlib as plt
import cv2
import open3d as o3d 

from threading import Timer
import paho.mqtt.client as mqtt
from PIL import Image # binary mask
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
NUM_TOTAL_CAMERAS = 1

received_frames_dict = {}
received_frames_lock = threading.Lock()
batch_timeout = 15 # seconds

# DECODE MSG. Params: file and name
def decode_file(enc_file, enc_name):
    ## RGB file decode  
    if 'color' in enc_name:
        decode_c_file_binary = base64.b64decode(enc_file)
        if len(decode_c_file_binary) % 2 != 0:
            decode_c_file_binary += b'\x00'
            logger.debug(f"Adjusted len color_image_data: {len(decode_c_file_binary)}\n")
        frame_decoded = cv2.imdecode(np.frombuffer(decode_c_file_binary, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame_decoded is None:
            logger.error("Decoded color image is None. Error loaf¡ding image or corverting color image to RGB. Check the data format")
    ## Depth file decode  
    if 'depth' in enc_name:
        decode_d_file_binary = base64.b64decode(enc_file)
        if len(decode_d_file_binary) % 2 != 0:
            decode_d_file_binary += b'\x00'
            logger.debug(f"Adjusted len depth_image_data: {len(decode_d_file_binary)}\n")
        frame_decoded = cv2.imdecode(np.frombuffer(decode_d_file_binary, dtype=np.uint16), cv2.IMREAD_UNCHANGED)
        if frame_decoded is None:
            logger.error("Decoded depth image is None, check the data format")

    return frame_decoded


# ENCODE MSG. Params: name # use to create de point cloud from encoded data
def encode_file(name_frame):
    ## Color file encode  
    if 'color' in name_frame:
        with open(name_frame, "rb") as color_file:
            encoded_string = base64.b64encode(color_file.read()).decode('utf-8')
    ## Depth file encode  
    if 'depth' in name_frame:
        with open(name_frame, "rb") as depth_file:
            encoded_string = base64.b64encode(depth_file.read()).decode('utf-8')

    return encoded_string


def create_pc_from_data(color_file_enc, color_name, depth_file_enc, depth_name, K, target_ds):
    color_image = decode_file(color_file_enc, color_name)
    depth_image = decode_file(depth_file_enc, depth_name)
    try:
        # Convert BGR to RGB .....(yuv to RGB (check))
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGR_I420)
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
    
    vox_size, _ = get_config(target_ds) # 0.00001 * 2.5
    logger.debug(f"[TEST] Voxel size: 0.00001")
    pcd = pcd.voxel_down_sample(voxel_size=0.00001)
    
    logger.debug(f"PCD len after downsampling {len(pcd.points)}")

    pcd.estimate_normals()

    return pcd

# path of the mask and the image encoded. Return: color image with the background set to black 
def apply_mask(mask_path, rgb_enc, color_name):
    # create binary mask  
    seg_img = cv2.imread(mask_path) # img: segmented image
    color_img= decode_file(rgb_enc, color_name) # img: color image
    high, width = seg_img.shape[:2]
    mask = np.zeros((high, width, 3), dtype='uint8')
    
    # Loop through each pixel of the image
    for m in range(seg_img.shape[0]):  # rows
        for n in range(seg_img.shape[1]):  # columns
            if np.array_equal(seg_img[m, n], [61, 5, 150]):  # cv2 BGR: instead of RGB; [61,5,150] people
                mask[m, n] = [255, 255, 255]  # Set to white
            else:
                mask[m, n] = [0, 0, 0]  # Set to black
    cv2.imwrite(mask_path, mask) # save binary mask; # overwrite the path of the segmented image

    # apply binary mask to the color frame 
    # Loop through each pixel to set to black the background
    for m in range(color_img.shape[0]):  # Iterate over rows
        for n in range(color_img.shape[1]):  
            if np.array_equal(mask[m, n], [255, 255, 255]):  # White pixel in the mask
                color_img[m, n] = color_img[m, n]  # the original
            else: 
                color_img[m, n] = 0  # background out 
    
    return color_img


# Dataset ADE20k: person index = 12 [0-149] 150 classes -> color = [150, 5, 61] 
def segmentator(color_frame_encod, color_frame_name, camera_name):
    # create folder to save segmented frames ------------------------------------
    mask_dict = 'images/mask'
    os.makedirs(mask_dict, exist_ok=True)
    img_path = os.path.join(mask_dict, color_frame_name)

    ## FIRST: parameters to initialize a segmentor ------------------------------
    config_file = 'configs/pspnet/pspnet_r50-d8_4xb4-80k_ade20k-512x512.py' # model PSPNET, dataset: ADE20k
    checkpoint_file = 'work_dir_el_weno/test_visual/iter_80000.pth' # the model is already trained 
    device_selected = 'cuda:7'  # OJO checkear antes 

    # model: the constructed segmentor
    model = init_model(config_file, checkpoint_file, device = device_selected)
    
    ## SECOND: inference image(s) with the sementor -----------------------------
    # note: images can be either image files (str) or loaded images (np.ndarray) 
    # result: np.array, the drawn image which channel is RGB or a dict [RGBimage, predictions]???????
    color_frame_dec = decode_file(color_frame_encod, color_frame_name)
    result = inference_model(model, color_frame_dec) 
    
    mask_frame_name = color_frame_name.replace('color', 'mask')
    out_file_model = os.path.join(mask_dict, mask_frame_name)

    # THIRD: save result --------------------------------------------------------
    #       opacity = 1. Full opaque --> color = [150, 5, 61] 
    #       with_labels = False. Dont show labels, not useful 
    # SAVE the results
    show_result_pyplot(
        model,
        img_path,
        result,
        title = None,
        opacity = 1,
        with_labels = False,
        draw_gt=False,
        show = False ,      # if args.out_file is not None else True,
        out_file = out_file_model,
    )

    logger.info(f"Model inference done. {os.path.basename(color_frame_name)} of CAMERA {camera_name}")
    return result, out_file_model 


# process frames of the same frame number 
def process_frames(msg_frames_list, num_frame):
    i = 1 # controlar 
    pcd_list = [] # deberia contener todas las pc de todas las camaras para un mismo frame, lista de 8 elementos

    for message in msg_frames_list:
        camera_name = message.get('folder_name')
        color_frame_name = message.get('frame_color_name')
        enc_c = message.get('enc_c')
        depth_frame_name = message.get('frame_depth_name')
        enc_d = message.get('enc_d')
        ts = message.get('send_ts')
        K = message.get('K') #[[915.76, 0.0, 965.475], [0.0, 916.205, 551.795], [0.0, 0.0, 1.0]] #
        target_ds = 1 # params if were exist more than one method or dataset ///// using now to not modify the rest of scripts 
    
        ## SECOND: preprocessing color frame before creating the pc 
        # segmentation and new rgb image 
        resutl, mask_path = segmentator(enc_c, color_frame_name, camera_name) # camera_name just to show info 
        if resutl is None: 
            logger.error(f"Error in preprocessing frame: {color_frame_name} [SEGMENTATION]. No segmented image")
        else:
            # args: path of the mask and the image encoded
            masked_img = apply_mask(mask_path, enc_c, color_frame_name)
        
        ## PRE-THIRD encode files (reading frames from pahts doesnt work [downsalmpling: ~1million points -> 3 points], create pc from encoding files works)
        enc_c = encode_file(masked_img)
        ## THIRD: create pc from each rgb image
        pc = create_pc_from_data(enc_c, color_frame_name, enc_d, depth_frame_name, K, target_ds)
        
        # no quitar de momento. al tener 1 sola camara se crea la nube de puntos de ese frame y ya  
        save_and_upload_pcd(pc, f"20250121_simple_pc_{camera_name}_{num_frame}.ply")
        
        pcd_list.append(pc)
        if pc is None:
            logger.error(f"Error creating point cloud for frame {color_frame_name} of {camera_name}")
            i += 1
            continue
        logger.info(f"[TS] Frame [{color_frame_name}] PCD created for camera {camera_name}")
        i += 1

    ## de momento no fusion pq solo tengo una camara
    ## FOURTH: fusion 
    # params if were exist more than one method or dataset ///// using now to not modify the rest of scripts 
    #target_registration = 0 # icp_p2p_registration_ransac   # MIKEL: message_json_list[0].get('reg')
    #final_fused_point_cloud = icp_p2p_registration_ransac(pcd_list, target_ds) 

    #if final_fused_point_cloud is None:
    #    logger.info(f"Frame [{num_frame}] Final PCD is None. Please check error logs.")
    #else:
    #    logger.debug(f"FRAME [{num_frame}] REGISTRATION SUCCESSFUL")
    #    
    #    reg_name = "icp_p2p_ransac" #reg_name = registration_names_from_id.get(target_registration, "unknown")
    #    dataset_name = "real" #dataset_name_from_id.get(target_ds, "unknown")
    #    blob_name_reg = f"20250117_reg_{num_frame}_{reg_name}_{dataset_name}.ply"
    #    
    #    save_and_upload_pcd(final_fused_point_cloud, blob_name_reg)
        

# si pasan los x segundos
def on_batch_timeout(num_frame):
    with received_frames_lock:
        logger.info(f"Timeout for frame {num_frame} detected")
        if num_frame in received_frames_dict and received_frames_dict[num_frame][0]:
            received_frames_dict[num_frame][1].cancel()  # Stop the timer
            frame_data_copy = received_frames_dict.pop(num_frame)[0]
            threading.Thread(target=process_frames, args=(frame_data_copy, num_frame)).start()


# process_message function extracts all information from the message received 
def process_message(msg):
    # get MAIN INFO OF THE FRAME FROM the payload: only camera name and number's frame 
    message = json.loads(msg.payload)
    camera_name = message.get('folder_name')
    color_frame_name = message.get('frame_color_name')
    _,_,num_frame = (color_frame_name.split('.')[0]).split('_') # name frame: cameraname_typeframe_numberframe.png
    logger.info(f"[TS] Received MSG from the camera {camera_name}, number: {num_frame}.")
    
    with received_frames_lock:
        # NEW frame, set the timer 
        if num_frame not in received_frames_dict: 
            received_frames_dict[num_frame] = ([], Timer(batch_timeout, on_batch_timeout, args=[num_frame]))
            received_frames_dict[num_frame][1].start() 
        
        received_frames_dict[num_frame][0].append(message) # save ALL frames; key: num_frame

        # when dict is complete for all frames of different cameras --> START  
        if len(received_frames_dict[num_frame][0]) == NUM_TOTAL_CAMERAS: 
            logger.info(f"Batch full for frame {num_frame}. START processing the frames")
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