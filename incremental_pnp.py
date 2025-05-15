import cv2
import numpy as np

def solve_PnP(self, obj_point, image_point, K, dist_coeff, rot_vector, initial) -> tuple:
    if obj_point is None or image_point is None or len(obj_point) < 4 or len(image_point) < 4:
        print("Warning: Not enough points for PnP!")
        return np.eye(3), np.zeros((3, 1)), image_point, obj_point, rot_vector

    try:
        obj_point = np.asarray(obj_point).reshape(-1, 3)
        image_point = np.asarray(image_point).reshape(-1, 2)

        obj_point = np.float64(obj_point)
        image_point = np.float64(image_point)
        K = np.float64(K)

        valid_mask = ~(np.any(np.isnan(obj_point), axis=1) | np.any(np.isinf(obj_point), axis=1) |
                    np.any(np.isnan(image_point), axis=1) | np.any(np.isinf(image_point), axis=1))

        obj_point = obj_point[valid_mask]
        image_point = image_point[valid_mask]

        if len(obj_point) < 4:
            print("Warning: Too few valid points for PnP after filtering!")
            return np.eye(3), np.zeros((3, 1)), image_point, obj_point, rot_vector

        success, rot_vector_calc, tran_vector, inliers = cv2.solvePnPRansac(
            obj_point, image_point, K, dist_coeff,
            flags=cv2.SOLVEPNP_ITERATIVE,
            iterationsCount=self.params['pnp_iterations'],
            reprojectionError=self.params['reproj_error'],
            confidence=self.params['pnp_confidence']
        )

        if not success:
            print("Warning: PnP failed to find a solution!")
            return np.eye(3), np.zeros((3, 1)), image_point, obj_point, rot_vector

        rot_matrix, _ = cv2.Rodrigues(rot_vector_calc)

        if inliers is not None and len(inliers) >= 4:
            inliers = inliers.ravel()
            image_point = image_point[inliers]
            obj_point = obj_point[inliers]
            if rot_vector.size > 0:
                rot_vector = rot_vector[inliers]

        return rot_matrix, tran_vector, image_point, obj_point, rot_vector

    except Exception as e:
        print(f"Error in solve_PnP: {str(e)}")
        return np.eye(3), np.zeros((3, 1)), image_point, obj_point, rot_vector