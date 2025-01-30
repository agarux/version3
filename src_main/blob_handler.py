from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import open3d as o3d
import os

#from constants import BLOB_CONNECTION_STRING, BLOB_CONTAINER_NAME
from logger_config import logger

# my account
BLOB_CONNECTION_STRING = 'DefaultEndpointsProtocol=https;AccountName=storagevisualizer;AccountKey=Sc1KrUTuRwgnCML86yVTtYJfhdI7osxXce4AM2hYaaXwgjDweR6NUHImawJXS7KToGf1HszobA8G+AStDsgvqw==;EndpointSuffix=core.windows.net'


blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
#container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)  

def save_and_upload_pcd(pcd, blob_name, container_name):
    save_point_cloud_to_ply(pcd, blob_name)
    if upload_ply_to_blob_storage(blob_name, container_name):
        os.remove(blob_name)
        logger.debug(f"{blob_name} removed from disk.")
    
def save_point_cloud_to_ply(point_cloud, file_path):
    try:
        o3d.io.write_point_cloud(file_path, point_cloud)
        logger.debug(f"Point cloud saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving point cloud to PLY: {e}")
        logger.debug(f"point_cloud class == {point_cloud.__class__}  file_path == {file_path.__class__}")
        return False

def upload_ply_to_blob_storage(blob_name, container_name):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
        # look for existing container if not create a new one 
        # get list of the names of the containers 
        containers = blob_service_client.list_containers() # container (data+metadata)
        containers_names = [container.name for container in containers] # name (metadata)
        if container_name not in containers_names: 
            container_client = blob_service_client.create_container(container_name) # create
        else:
            container_client = blob_service_client.get_container_client(container_name)
        
        blob_client = container_client.get_blob_client(blob_name)
        
        with open(blob_name, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        logger.info(f"[TS] {blob_name} uploaded to Azure Blob Storage.")
        return True
    except Exception as e:
        logger.error(f"Error uploading file to Blob Storage: {e}")
        return False