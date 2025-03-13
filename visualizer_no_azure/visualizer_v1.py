import os
import threading
from werkzeug.exceptions import HTTPException
from flask import Flask, jsonify, request, render_template
import open3d as o3d
import time
import numpy as np

# Define the path to the local folder where point cloud files are stored
LOCAL_FOLDER_PATH = "./seq_c2"  # Change this to the path of your folder containing point clouds

FRAME_RATE = 1/30

# Initialize Flask application
app = Flask(__name__)

# Enable WebRTC for Open3D
o3d.visualization.webrtc_server.enable_webrtc()

# AUX functions 
def rotate_point_cloud(pc):
    angle_rad = np.radians(180) 
    R = pc.get_rotation_matrix_from_xyz((0, angle_rad, angle_rad))
    pc.rotate(R)
    return pc


def show_blob_o3d(chosen_pointsize, chosen_framerate, chosen_background, point_clouds):
    # Visualization 
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='PointCloud visualizer', height=540, width=960, left = 400, top = 300)
    vis.get_render_option().background_color = chosen_background
    vis.get_render_option().point_size = float(chosen_pointsize)
    chosen_framerate = 1/int(chosen_framerate)

    try:
        current_index = 0 
        if point_clouds:
            vis.add_geometry(point_clouds[current_index]) 
        while True:
            vis.remove_geometry(point_clouds[current_index], reset_bounding_box=False)  # False to keep the current viewpoint 
            current_index = (current_index + 1) % len(point_clouds) 
            current_point_cloud = point_clouds[current_index]
            vis.add_geometry(current_point_cloud, reset_bounding_box=False)  # False to keep the current viewpoint 

            vis.update_geometry(current_point_cloud)
            vis.update_renderer()

            time.sleep(chosen_framerate) 
            if not vis.poll_events():
                break            
    except KeyboardInterrupt:
        print("Closing window...")
    finally:
        vis.destroy_window()


# Define route() decorators to bind a function to a URL 
@app.route("/", methods=['POST', 'GET']) 
def view_container(): 
    return render_template('index.html')  # Main form page

# List all available files in the local folder (instead of Azure containers)  
@app.route("/get_files", methods=['GET'])
def get_files():
    try:
        files = [f for f in os.listdir(LOCAL_FOLDER_PATH) if f.endswith('.ply') or f.endswith('.pcd')]  # Adjust extensions as needed
    except Exception as e:
        print(e)
        files = []
    return jsonify(files)


@app.route("/back_to_main", methods=['POST', 'GET'])
def back_to_main():
    return render_template('index.html')


@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return e
    return render_template('error_template.html', e=e), 500


@app.route("/get_visualization", methods=['POST', 'GET'])
def get_visualization():
    if request.method == 'POST':
        try:
            #chosen_folder = request.form.get('folder_name')
            chosen_pointsize = request.form.get('point_size')
            chosen_framerate = request.form.get('fps_rate')
            chosen_background = request.form.get('color_bkg')
            if chosen_background == 'black':
                chosen_background = [0, 0, 0]
            if chosen_background == 'white':
                chosen_background = [255, 255, 255]
        except Exception as e:
            return "Please provide all parameters."

        # Read point cloud files from the local folder
        point_clouds = []
        for filename in os.listdir(os.path.join(LOCAL_FOLDER_PATH)):
            if filename.endswith('.ply') or filename.endswith('.pcd'):  # Read only supported files
                try:
                    file_path = os.path.join(LOCAL_FOLDER_PATH, filename)
                    point_cloud = o3d.io.read_point_cloud(file_path)
                    rotate_pc = rotate_point_cloud(point_cloud)
                    if not rotate_pc.is_empty():
                        point_clouds.append(rotate_pc)
                    else:
                        print(f"ERROR: {filename} is empty or invalid.")
                except Exception as e:
                    print(f"ERROR reading {filename}: {e}")

        # Launch visualizer in a different thread to avoid blocking the Flask main thread
        thread = threading.Thread(target=show_blob_o3d, args=(chosen_pointsize, chosen_framerate, chosen_background, point_clouds))
        thread.start()
        return render_template('second_temp.html')  # Redirect to a page showing the visualization


if __name__ == "__main__":
    app.run(debug=False, port=5002)
