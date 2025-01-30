## New visualizer Open3d igual que visualizer_v1 pero "mas interactivo" 
# Jan 2025
# connect to Azure Blob Storage 
# display all blob in order 
# Source : tfm mikel using his azure functions  
#          https://learn.microsoft.com/es-es/azure/storage/blobs/storage-blob-python-get-started?tabs=azure-ad
#          https://stackoverflow.com/questions/65774814/adding-new-points-to-point-cloud-in-real-time-open3d
#          https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-container-create-python 
# contiene las clases principales (objetos de cliente) que puede usar para operar een el servicio, los contenedores y los blobs
# https://www.geeksforgeeks.org/image-and-video-storage-with-azure-blob-storage-media-applications/

import io
import os
from azure.storage.blob import BlobServiceClient
from flask import Flask, jsonify, request, redirect, render_template, send_file
from mimetypes import guess_type
import open3d as o3d
import time 

BLOB_CONNECTION_STRING = 
FRAME_RATE = 1/30

# initialize flask application
app = Flask(__name__)

# Enable WebRTC for Open3D
o3d.visualization.webrtc_server.enable_webrtc()

# initiate a connection to AzureBlobStorage with connection string 
blob_service_client = BlobServiceClient.from_connection_string(conn_str=BLOB_CONNECTION_STRING)
try:
	containers = blob_service_client.list_containers()
except Exception as e:
	print(f"Error listing containers of the Azure Blob Storage. Error: {e}")


# define routes 
@app.route("/") # displays the options in the selectable part. AJAX --> dinamically uploading the options
def index():
	return render_template('index.html') # File saved in src_azurefun/templates/index.html

# list all avaliable cointainers in connection string  
@app.route("/get_containers", methods=['GET'])
def get_containers():
	containers_name = []
	try:
		containers_items = blob_service_client.list_containers()
		containers_name = [container.name for container in containers_items]
	except Exception as e:
		print(e)
	return jsonify (containers_name)


if __name__ == "__main__":
	app.run(debug=False, port = 5002)
