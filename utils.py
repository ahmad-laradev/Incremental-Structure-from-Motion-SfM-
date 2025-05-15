import cv2
import numpy as np
from scipy.spatial import cKDTree
import os

def find_common_points(self, image_points_1, image_points_2, image_points_3) -> tuple:
    pts1 = np.asarray(image_points_1, dtype=np.float32)
    pts2 = np.asarray(image_points_2, dtype=np.float32)

    cm_points_1 = []
    cm_points_2 = []

    pts1 = pts1.reshape(-1, 1, 2)
    pts2 = pts2.reshape(1, -1, 2)

    distances = np.sqrt(np.sum((pts1 - pts2) ** 2, axis=2))
    threshold = 1.0
    matches = distances < threshold

    for i in range(len(pts1)):
        potential_matches = np.where(matches[i])[0]
        if len(potential_matches) > 0:
            best_match = potential_matches[np.argmin(distances[i][potential_matches])]
            cm_points_1.append(i)
            cm_points_2.append(best_match)

    cm_points_1 = np.array(cm_points_1, dtype=np.int32)
    cm_points_2 = np.array(cm_points_2, dtype=np.int32)

    if len(cm_points_2) > 0:
        mask = np.ones(len(image_points_2), dtype=bool)
        mask[cm_points_2] = False
        mask_array_1 = image_points_2[mask]
        mask_array_2 = image_points_3[mask]
    else:
        mask_array_1 = image_points_2
        mask_array_2 = image_points_3

    print(f"Found {len(cm_points_1)} common points between frames")
    print(f"Remaining points after filtering: {len(mask_array_1)}")
    return cm_points_1, cm_points_2, mask_array_1, mask_array_2

def reproj_error(self, obj_points, image_points, transform_matrix, K, homogenity) -> tuple:
    if obj_points.size == 0 or image_points.size == 0:
        print("Warning: Empty points array in reproj_error")
        return float('inf'), obj_points

    try:
        if homogenity == 1:
            if obj_points.shape[0] == 4:
                obj_points = obj_points.T
            elif len(obj_points.shape) == 3:
                obj_points = obj_points.reshape(-1, 4)

            try:
                obj_points = cv2.convertPointsFromHomogeneous(obj_points)
                obj_points = obj_points.reshape(-1, 3)
            except cv2.error as e:
                print(f"Error in homogeneous conversion: {str(e)}")
                return float('inf'), obj_points

        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)

        if len(obj_points.shape) == 3:
            obj_points = obj_points.reshape(-1, 3)

        image_points_calc, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points_calc = np.float32(image_points_calc[:, 0, :])

        if homogenity == 1:
            image_points = np.float32(image_points.T)
        else:
            image_points = np.float32(image_points)

        total_error = cv2.norm(image_points_calc, image_points, cv2.NORM_L2)
        return total_error / len(image_points_calc), obj_points

    except Exception as e:
        print(f"Error in reproj_error: {str(e)}")
        return float('inf'), obj_points
import os
import numpy as np
from sklearn.cluster import DBSCAN


def save_to_ply(self, path, point_cloud, colors=None, bundle_adjustment_enabled=False,
               binary_format=False, scaling_factor=1.0):
    # Get the actual output directory
    base_dir = os.path.dirname(os.path.abspath(__file__)) if not os.path.isabs(path) else path
    sub_dir = 'Results with Bundle Adjustment' if bundle_adjustment_enabled else 'Results'
    output_dir = os.path.join(base_dir, sub_dir)

    # If 'Results' folder doesn't exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filtered_ply_path = os.path.join(output_dir, "point_cloud.ply")  # Only save the filtered.ply file

    point_cloud = np.float32(point_cloud)
    colors = np.uint8(colors) if colors is not None else np.zeros((len(point_cloud), 3), dtype=np.uint8)

    # Initial filtering: Remove NaNs and Infs
    valid_mask = ~np.any(np.isnan(point_cloud), axis=1) & ~np.any(np.isinf(point_cloud), axis=1)
    point_cloud = point_cloud[valid_mask]
    colors = colors[valid_mask]

    # Remove points with all zeros
    non_zero_mask = ~np.all(np.abs(point_cloud) < 1e-6, axis=1)
    point_cloud = point_cloud[non_zero_mask]
    colors = colors[non_zero_mask]

    if len(point_cloud) == 0:
        print("Error: No valid points")
        return

    # Statistical filtering
    mean = np.mean(point_cloud, axis=0)
    std = np.std(point_cloud, axis=0)
    inlier_mask = np.all(np.abs(point_cloud - mean) < 2 * std, axis=1)
    point_cloud = point_cloud[inlier_mask]
    colors = colors[inlier_mask]

    # Apply bundle adjustment or MAD filtering
    if bundle_adjustment_enabled:
        # DBSCAN clustering for bundle-adjusted results
        normalized_points = (point_cloud - np.mean(point_cloud, axis=0)) / np.std(point_cloud, axis=0)
        db = DBSCAN(eps=0.35, min_samples=15).fit(normalized_points)
        labels = db.labels_

        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(unique_labels) > 0:
            main_cluster = unique_labels[np.argmax(counts)]
            cluster_mask = (labels == main_cluster)
        else:
            cluster_mask = np.ones_like(labels, dtype=bool)

        point_cloud = point_cloud[cluster_mask]
        colors = colors[cluster_mask]
    else:
        # MAD filtering for non-bundle-adjusted results
        centroid = np.median(point_cloud, axis=0)
        distances = np.linalg.norm(point_cloud - centroid, axis=1)
        median_dist = np.median(distances)
        mad = np.median(np.abs(distances - median_dist))
        threshold = median_dist + 3 * mad
        mad_mask = distances <= threshold
        point_cloud = point_cloud[mad_mask]
        colors = colors[mad_mask]

    # Final scaling and centering
    centroid = np.mean(point_cloud, axis=0)
    point_cloud -= centroid
    max_distance = np.max(np.linalg.norm(point_cloud, axis=1))
    if max_distance > 0:
        point_cloud *= (scaling_factor / max_distance)

    print(f"Final point count: {len(point_cloud)}")

    # Save only filtered results (no main .ply file)
    with open(filtered_ply_path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(point_cloud)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for i in range(len(point_cloud)):
            x, y, z = point_cloud[i]
            r, g, b = colors[i]
            f.write(f'{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n')

    print(f"Saved filtered point cloud with {len(point_cloud)} points to {filtered_ply_path}")
