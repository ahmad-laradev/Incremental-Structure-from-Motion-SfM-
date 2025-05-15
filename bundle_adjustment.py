import numpy as np
import cv2
from scipy.optimize import least_squares

def optimize_reproj_error(self, obj_points) -> np.array:
    transform_matrix = obj_points[0:12].reshape((3, 4))
    K = obj_points[12:21].reshape((3, 3))
    rest = int(len(obj_points[21:]) * 0.4)
    p = obj_points[21:21 + rest].reshape((2, int(rest / 2))).T
    obj_points = obj_points[21 + rest:].reshape((int(len(obj_points[21 + rest:]) / 3), 3))

    rot_matrix = transform_matrix[:3, :3]
    tran_vector = transform_matrix[:3, 3]
    rot_vector, _ = cv2.Rodrigues(rot_matrix)

    image_points, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
    image_points = image_points[:, 0, :]

    error = [(p[idx] - image_points[idx]) ** 2 for idx in range(len(p))]
    return np.array(error).ravel() / len(p)

def compute_bundle_adjustment(self, _3d_point, opt, transform_matrix_new, K, r_error) -> tuple:
    opt_variables = np.hstack((transform_matrix_new.ravel(), K.ravel()))
    opt_variables = np.hstack((opt_variables, opt.ravel()))
    opt_variables = np.hstack((opt_variables, _3d_point.ravel()))

    values_corrected = least_squares(self.optimize_reproj_error, opt_variables, gtol=r_error).x

    K = values_corrected[12:21].reshape((3, 3))
    rest = int(len(values_corrected[21:]) * 0.4)
    return (values_corrected[21 + rest:].reshape((int(len(values_corrected[21 + rest:]) / 3), 3)),
            values_corrected[21:21 + rest].reshape((2, int(rest / 2))).T,
            values_corrected[0:12].reshape((3, 4)))