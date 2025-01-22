## New visualizer Open3d 
# Jan 2025
# connect to Azure Blob Storage 
# display all blob in order 
# Source : tfm mikel using his azure functions  
#          https://learn.microsoft.com/es-es/azure/storage/blobs/storage-blob-python-get-started?tabs=azure-ad
#          https://stackoverflow.com/questions/65774814/adding-new-points-to-point-cloud-in-real-time-open3d

# contiene las clases principales (objetos de cliente) que puede usar para operar een el servicio, los contenedores y los blobs
import io
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import numpy as np
import open3d as o3d
import time
import logging


FRAME_RATE = 0.5

def download_blob(blob_client, target_blob):
    with open(target_blob, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    logging.info(f"Downloaded blob: {target_blob}")
    return target_blob


def main():
    # Connection to Azure Blob Storage get a CLIENT object and the container
    blob_service_client = BlobServiceClient.from_connection_string(conn_str=BLOB_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
    # all elements no duplications with data and metadata
    blob_list = container_client.list_blobs()
    # saved only data 
    point_clouds = []

    for blob in blob_list:
        print(f"Processing {blob.name}")
        # Get blob client and download the blob content
        blob_client = container_client.get_blob_client(blob)

        filename = download_blob(blob_client, blob.name)
        # añadir nube de puntos a una lista 
        if filename:
            try:
                point_cloud = o3d.io.read_point_cloud(filename)
                if not point_cloud.is_empty():
                    point_clouds.append(point_cloud)
                else:
                    print(f"Blob {blob.name} ERROR. Invalid data")
            except Exception as e:
                print(f"ERROR {blob.name}: {e}")
        else:
            print("No hay nube")

    print("Nubes cargadas ")
    # initialize el visualizer open3d
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=540, width=810)

    # visualization 
    if point_clouds:
        # add pc to the window
        for pc in point_clouds:
            vis.add_geometry(pc)
            time.sleep(FRAME_RATE)
            
        # update window 
        vis.update_geometry(pc)
        vis.poll_events()
        vis.run()
        vis.destroy_window()
        print(f"Loaded point cloud from")
        
    else:
        print("No point clouds found to visualize.")
    
    

main()