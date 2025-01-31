## New visualizer Open3d 
# Jan 2025
# connect to Azure Blob Storage 
# display all blobs in order 
# Source : tfm mikel using his azure functions  
#          https://learn.microsoft.com/es-es/azure/storage/blobs/storage-blob-python-get-started?tabs=azure-ad
#          https://stackoverflow.com/questions/65774814/adding-new-points-to-point-cloud-in-real-time-open3d
#          https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-container-create-python 
# contiene las clases principales (objetos de cliente) que puede usar para operar een el servicio, los contenedores y los blobs

######
# V1 # datos por la terminal 
######

import io
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import numpy as np
import open3d as o3d
import time
import os

BLOB_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=storagevisualizer;AccountKey=Sc1KrUTuRwgnCML86yVTtYJfhdI7osxXce4AM2hYaaXwgjDweR6NUHImawJXS7KToGf1HszobA8G+AStDsgvqw==;EndpointSuffix=core.windows.net"
FRAME_RATE = 1/30

def download_blob(blob_client, target_blob):
    if os.path.isfile(target_blob):
        print("No se descargo del azure")
        return target_blob
    else:
        with open(target_blob, "wb") as download_file:
            print("Descargando del azure...")
            download_file.write(blob_client.download_blob().readall())
            return target_blob
    

def rotate_point_cloud(pc):
    angle_rad = np.radians(180) 
    R = pc.get_rotation_matrix_from_xyz((0, angle_rad, angle_rad))
    pc.rotate(R)
    return pc


# download data from Azure Storage or read downloaded data
def get_point_clouds(blob_service_client, name_container):
    container_client = blob_service_client.get_container_client(name_container)
    # all elements no duplications with data and metadata
    blob_list = container_client.list_blobs()
    # saved only data 
    point_clouds = []
    print("Loading frames (wait)........................................")
    for blob in blob_list:
        # Get blob client and download the blob content
        blob_client = container_client.get_blob_client(blob)
        filename = download_blob(blob_client, blob.name)
        # añadir nube de puntos a una lista 
        if filename:
            try:
                point_cloud = o3d.io.read_point_cloud(filename)
                rotate_pc = rotate_point_cloud(point_cloud)
                if not rotate_pc.is_empty():
                    point_clouds.append(rotate_pc)
                else:
                    print(f"Blob {blob.name} ERROR. Invalid data")
            except Exception as e:
                print(f"ERROR {blob.name}: {e}")
        else:
            print("ERROR. File found but there isnt data")

    print("Frames loaded!")
    return point_clouds

# Open3d
def result(point_clouds, point_size, fps):
    fps = 1/fps
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='PointCloud visualizer', height=540, width=960)
    vis.get_render_option().background_color = [255, 255, 255]
    vis.get_render_option().point_size = float(point_size)

    try:
        current_index = 0 
        if point_clouds:
            vis.add_geometry(point_clouds[current_index]) 
        while True:
            vis.remove_geometry(point_clouds[current_index], reset_bounding_box = False)  # false to keep current viewpoint 
            current_index = (current_index + 1) % len(point_clouds) 
            current_point_cloud = point_clouds[current_index]
            vis.add_geometry(current_point_cloud, reset_bounding_box = False)  # false to keep current viewpoint 

            vis.update_geometry(current_point_cloud)
            vis.update_renderer()

            time.sleep(fps) 
            if not vis.poll_events():
                break            
    except KeyboardInterrupt:
        print("Closing window...")
    finally:
        vis.destroy_window()
        print("END")

# poner datos por defecto en caso de introducirlos mal o simplemente porner el try:except
def data_selection_prompt():
    print("Select the size of the points (0.0 - 5.0): ")
    ps = input("Selection: ")
    print("Select the sequence: depth (0), media (1), new29 (2): ")
    seq = input("Selected number:")
    seq = int(seq)
    print("Velocidad de reproduccion en FPS (30FPS, 15FPS...): ")
    fps = input("Selected FPS number:")
    fps = int(fps)
    if seq == 0:
        name_container = "depth"
    if seq == 1:
        name_container = "media"
    if seq == 2:
        name_container = "new29"
    return name_container, ps, fps

def main():
    name_container, point_size, fps = data_selection_prompt()
    # Connection to Azure Blob Storage get a CLIENT object and the container
    blob_service_client = BlobServiceClient.from_connection_string(conn_str=BLOB_CONNECTION_STRING)
    # get data to visualize 
    point_clouds = get_point_clouds(blob_service_client, name_container)
    # initialize el visualizer open3d, fondo negro y point_size determinado 
    result(point_clouds, point_size, fps)

    
main()
