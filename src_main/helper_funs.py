import numpy as np
from logger_config import logger
import os

def get_config(id_dataset):
    logger.debug(f"ID DATASET: {id_dataset}")
    if id_dataset == 0: #3dmatch
        BACKBONE_INIT_VOXEL_SIZE = 0.025
        threshold = 0.01
    elif id_dataset == 1: #own data
        BACKBONE_INIT_VOXEL_SIZE = 0.00001 * 2.5
        threshold = 0.0025
        # threshold = 1e-05
    else: # use 3dmatch as default
        BACKBONE_INIT_VOXEL_SIZE = 0.025
        threshold = 0.01
    return BACKBONE_INIT_VOXEL_SIZE, threshold


def calculate_registration_metrics(pc_source, pc_ref, threshold, p2p_registration=False, final_result_print=False, i=1):
    running_env = os.getenv('ENV', 'test')
    if running_env != 'test':
        logger.info(f"ENVIRONMENT is not test. Skipping metrics.")
        return
    
    """
    This function calculates the Root Mean Squared Error (RMSE), Inlier Ratio (IR), Recall Ratio (RR), 
    and False Match Ratio (FMR) between two point clouds represented by Open3d PointCloud objects.

    Args:
        source_pc: An open3d.cpu.pybind.geometry.PointCloud representing the source point cloud.
        ref_pc: An open3d.cpu.pybind.geometry.PointCloud representing the reference point cloud.

    Returns:
        A dictionary containing the following keys:
            rmse: The root mean squared error between the two point clouds.
            inlier_ratio: The ratio of points in the source cloud with a close correspondence in the reference cloud.
            recall_ratio: The ratio of points in the reference cloud that have a close correspondence in the source cloud.
            false_match_ratio: The ratio of points in the source cloud that do not have a close correspondence in the reference cloud.
    """
    if final_result_print:
        logger.info(f"\n########## REGISTERED FINAL POINT CLOUD METRICS ############")
    else:
        logger.info(f"\n############## ITERATION {i} METRICS ########################")
    
    # Convert point cloud coordinates to numpy arrays
    source_points = np.asarray(pc_source.points)
    ref_points = np.asarray(pc_ref.points)

    # Find the nearest neighbors for each point in the source point cloud
    nearest_neighbors = np.linalg.norm(source_points[:, np.newaxis] - ref_points, axis=2).argmin(axis=1)

    # Calculate distances between points and their nearest neighbors
    distances = np.linalg.norm(source_points - ref_points[nearest_neighbors], axis=1)

    # Define a threshold for considering a point a close correspondence
    # threshold = 0.1  # You can adjust this value based on your application

    # Count the number of inliers (points with distance below the threshold)
    inliers = np.count_nonzero(distances <= threshold)

    # Calculate RMSE
    rmse = np.sqrt(np.mean(distances**2))

    # Calculate Inlier Ratio (IR)
    inlier_ratio = inliers / len(source_points)

    # Calculate Recall Ratio (RR) - ratio of inliers in reference cloud
    recall_ratio = inliers / len(ref_points)

    # Calculate False Match Ratio (FMR) - ratio of outliers in source cloud
    false_match_ratio = 1 - inlier_ratio

    logger.info(
          f"### threshold: {threshold}"
          f"### RMSE: {rmse}"
          f"\n### Inlier Ratio: {inlier_ratio}"
          f"\n### Recall Ratio: {recall_ratio}"
          f"\n### False Match Ratio: {false_match_ratio}")
    # # Return the results as a dictionary
    # return {
    #     "rmse": rmse,
    #     "inlier_ratio": inlier_ratio,
    #     "recall_ratio": recall_ratio,
    #     "false_match_ratio": false_match_ratio
    # }
