## New visualizer Open3d 
# Jan 2025
# connect to Azure Blob Storage 
# display all blob in order 
# Source : tfm mikel using his azure functions +- 
#          https://learn.microsoft.com/es-es/azure/storage/blobs/storage-blob-python-get-started?tabs=azure-ad
#          https://stackoverflow.com/questions/65774814/adding-new-points-to-point-cloud-in-real-time-open3d
#          https://www.open3d.org/docs/latest/python_api/open3d.visualization.Visualizer.html
# contiene las clases principales (objetos de cliente) que puede usar para operar een el servicio, los contenedores y los blobs
import io
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import numpy as np
import open3d as o3d
import time

BLOB_CONNECTION_STRING = 
BLOB_CONTAINER_NAME = 
FRAME_RATE = 1/15 

def download_blob(blob_client, target_blob):
    with open(target_blob, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    return target_blob


def rotate_point_cloud(pc):
    angle_rad = np.radians(180) 
    R = pc.get_rotation_matrix_from_xyz((0, angle_rad, angle_rad))
    pc.rotate(R)
    return pc


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
                rotate_pc = rotate_point_cloud(point_cloud)
                if not rotate_pc.is_empty():
                    point_clouds.append(rotate_pc)
                else:
                    print(f"Blob {blob.name} ERROR. Invalid data")
            except Exception as e:
                print(f"ERROR {blob.name}: {e}")
        else:
            print("No hay nube")
    
    # initialize el visualizer open3d, fondo negro y point_size determinado 
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='PointCloud visualizer', height=540, width=810)
    vis.get_render_option().background_color = [0, 0, 0]
    vis.get_render_option().point_size = 2
    #vis.run()
    try:
        current_index = 0 
        if point_clouds:
            vis.add_geometry(point_clouds[current_index]) #), reset_bounding_box=False) # false to keep current viewpoint 
        while True:
            vis.remove_geometry(point_clouds[current_index], reset_bounding_box = False)  
            current_index = (current_index + 1) % len(point_clouds) 
            current_point_cloud = point_clouds[current_index]
            vis.add_geometry(current_point_cloud, reset_bounding_box = False) 

            vis.update_geometry(current_point_cloud)#,reset_bounding_box = False) 
            vis.poll_events()
            vis.update_renderer()
            
            time.sleep(FRAME_RATE)  
            
    except KeyboardInterrupt:
        print("Cerrando la ventana...")
    finally:
        vis.destroy_window()
        print("Fin del visualizador")
        
main()
