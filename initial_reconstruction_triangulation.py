import cv2
import numpy as np

def triangulation(self, proj_matrix_1, proj_matrix_2, pts_2d_1, pts_2d_2) -> tuple:
    if len(pts_2d_1) == 0 or len(pts_2d_2) == 0:
        print("Warning: No points to triangulate!")
        return np.array([]), np.array([]), np.array([])

    try:
        pts_2d_1 = np.asarray(pts_2d_1, dtype=np.float64)
        pts_2d_2 = np.asarray(pts_2d_2, dtype=np.float64)

        if len(pts_2d_1.shape) == 1:
            pts_2d_1 = pts_2d_1.reshape(-1, 2)
        if len(pts_2d_2.shape) == 1:
            pts_2d_2 = pts_2d_2.reshape(-1, 2)

        point_cloud = cv2.triangulatePoints(
            proj_matrix_1.astype(np.float64),
            proj_matrix_2.astype(np.float64),
            pts_2d_1.T.astype(np.float64),
            pts_2d_2.T.astype(np.float64)
        )

        point_cloud /= point_cloud[3]
        return pts_2d_1.T, pts_2d_2.T, point_cloud

    except Exception as e:
        print(f"Error in triangulation: {str(e)}")
        return np.array([]), np.array([]), np.array([])