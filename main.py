import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
from tqdm import tqdm
from camera_calibration import ImageLoader
from feature_detection_matching import feature_matching
from initial_reconstruction_triangulation import triangulation
from incremental_pnp import solve_PnP
from bundle_adjustment import compute_bundle_adjustment, optimize_reproj_error
from utils import find_common_points, reproj_error, save_to_ply

class StructurefromMotion:
    def __init__(self, img_dir=str, downscale_factor: float = 1.0, **kwargs):
        self.params = {
            'sift_features': 50000,
            'sift_contrast': 0.005,
            'ratio_thresh': 0.8,
            'essential_prob': 0.999,
            'essential_thresh': 0.4,
            'min_matches': 8,
            'reproj_error': 10.0,
            'pnp_confidence': 0.99,
            'pnp_iterations': 200,
            'max_points': 5000,
            'scaling_factor': 5000.0
        }
        self.params.update(kwargs)
        self.img_obj = ImageLoader(img_dir, downscale_factor)
    
    # Assign methods from respective files
    feature_matching = feature_matching
    triangulation = triangulation
    solve_PnP = solve_PnP
    compute_bundle_adjustment = compute_bundle_adjustment
    optimize_reproj_error = optimize_reproj_error
    find_common_points = find_common_points
    reproj_error = reproj_error
    save_to_ply = save_to_ply

    def __call__(self, bundle_adjustment_enabled: bool = False):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pose_array = self.img_obj.K.ravel()
        transform_matrix_0 = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0]])
        transform_matrix_1 = np.empty((3, 4))
        print('Camera Intrinsic Matrix:', self.img_obj.K)

        pose_0 = np.matmul(self.img_obj.K, transform_matrix_0)
        pose_1 = np.empty((3, 4))
        total_points = np.zeros((1, 3))
        total_colors = np.zeros((1, 3))

        image_0 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[0]))
        image_1 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[1]))

        features_0, features_1 = self.feature_matching(image_0, image_1)
        if len(features_0) == 0 or len(features_1) == 0:
            print("Skipping first pair due to no feature matches.")
            return

        essential_matrix, em_mask = cv2.findEssentialMat(
            features_0, features_1,
            self.img_obj.K,
            method=cv2.RANSAC,
            prob=self.params['essential_prob'],
            threshold=self.params['essential_thresh']
        )
        features_0 = features_0[em_mask.ravel() == 1]
        features_1 = features_1[em_mask.ravel() == 1]

        if len(features_0) == 0 or len(features_1) == 0:
            print("Skipping first pair due to no inliers after essential matrix filtering.")
            return

        _, rot_matrix, tran_matrix, em_mask = cv2.recoverPose(essential_matrix, features_0, features_1, self.img_obj.K)
        features_0 = features_0[em_mask.ravel() > 0]
        features_1 = features_1[em_mask.ravel() > 0]

        if len(features_0) == 0 or len(features_1) == 0:
            print("Skipping first pair due to no inliers after recoverPose.")
            return

        transform_matrix_1[:3, :3] = np.matmul(rot_matrix, transform_matrix_0[:3, :3])
        transform_matrix_1[:3, 3] = transform_matrix_0[:3, 3] + np.matmul(transform_matrix_0[:3, :3], tran_matrix.ravel())

        pose_1 = np.matmul(self.img_obj.K, transform_matrix_1)

        features_0, features_1, points_3d = self.triangulation(pose_0, pose_1, features_0, features_1)
        if len(points_3d) == 0:
            print("Skipping first pair due to failed triangulation.")
            return

        error, points_3d = self.reproj_error(points_3d, features_1, transform_matrix_1, self.img_obj.K, homogenity=1)
        print("Reprojection error for first two images:", error)

        _, _, features_1, points_3d, _ = self.solve_PnP(points_3d, features_1, self.img_obj.K,
                                                       np.zeros((5, 1), dtype=np.float32), features_0, initial=1)

        total_images = len(self.img_obj.image_list) - 2
        print('total_images', total_images)

        pose_array = np.hstack((np.hstack((pose_array, pose_0.ravel())), pose_1.ravel()))

        threshold = 0.75

        for i in tqdm(range(total_images)):
            image_2 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[i + 2]))

            features_cur, features_2 = self.feature_matching(image_1, image_2)
            if len(features_cur) == 0 or len(features_2) == 0:
                print(f"Skipping image index {i+2} due to no feature matches.")
                continue

            if i != 0:
                features_0, features_1, points_3d = self.triangulation(pose_0, pose_1, features_0, features_1)
                if len(points_3d) == 0:
                    print(f"Skipping iteration {i} due to failed triangulation.")
                    continue
                features_1 = features_1.T
                points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)

            cm_points_0, cm_points_1, cm_mask_0, cm_mask_1 = self.find_common_points(features_1, features_cur, features_2)
            if len(cm_points_0) == 0 or len(cm_points_1) == 0:
                print(f"Skipping iteration {i} due to no common points found.")
                continue

            cm_points_2 = features_2[cm_points_1]
            cm_points_cur = features_cur[cm_points_1]

            rot_matrix, tran_matrix, cm_points_2, points_3d, cm_points_cur = self.solve_PnP(points_3d[cm_points_0], cm_points_2, self.img_obj.K,
                                                                                          np.zeros((5, 1), dtype=np.float32), cm_points_cur, initial=0)
            if rot_matrix is None or tran_matrix is None:
                print(f"Skipping iteration {i} due to failed PnP.")
                continue

            transform_matrix_1 = np.hstack((rot_matrix, tran_matrix))
            pose_2 = np.matmul(self.img_obj.K, transform_matrix_1)

            error, points_3d = self.reproj_error(points_3d, cm_points_2, transform_matrix_1, self.img_obj.K, homogenity=0)
            if np.isinf(error) or np.isnan(error):
                print(f"Skipping iteration {i} due to invalid reprojection error.")
                continue

            cm_mask_0, cm_mask_1, points_3d = self.triangulation(pose_1, pose_2, cm_mask_0, cm_mask_1)
            if len(points_3d) == 0:
                print(f"Skipping iteration {i} due to failed triangulation.")
                continue

            error, points_3d = self.reproj_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K, homogenity=1)
            print("Reprojection error:", error)
            pose_array = np.hstack((pose_array, pose_2.ravel()))

            if bundle_adjustment_enabled:
                points_3d, cm_mask_1, transform_matrix_1 = self.compute_bundle_adjustment(points_3d, cm_mask_1,
                                                                                          transform_matrix_1, self.img_obj.K,
                                                                                          threshold)
                pose_2 = np.matmul(self.img_obj.K, transform_matrix_1)
                error, points_3d = self.reproj_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K, homogenity=0)
                print("Reprojection error after Bundle Adjustment: ", error)

            try:
                if len(points_3d.shape) == 3:
                    points_to_add = points_3d.reshape(-1, 3)
                elif points_3d.shape[0] == 4:
                    points_to_add = cv2.convertPointsFromHomogeneous(points_3d.T).reshape(-1, 3)
                else:
                    points_to_add = points_3d.reshape(-1, 3)

                points_2d = np.float32(cm_mask_1).reshape(-1, 2)

                if points_to_add.shape[1] == 3 and len(points_2d) > 0:
                    valid_mask = ((points_2d[:, 0] >= 0) & (points_2d[:, 0] < image_2.shape[1]) &
                                  (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image_2.shape[0]))

                    if np.any(valid_mask):
                        points_2d = points_2d[valid_mask]
                        points_to_add = points_to_add[valid_mask]
                        color_vector = np.array([image_2[int(y), int(x)] for x, y in points_2d])

                        if len(points_to_add) > 0:
                            total_points = np.vstack((total_points, points_to_add))
                            total_colors = np.vstack((total_colors, color_vector))
            except Exception as e:
                print(f"Warning: Error processing points: {str(e)}")

            transform_matrix_0 = np.copy(transform_matrix_1)
            pose_0 = np.copy(pose_1)
            plt.scatter(i, error)
            plt.pause(0.05)

            image_0 = np.copy(image_1)
            image_1 = np.copy(image_2)
            features_0 = np.copy(features_cur)
            features_1 = np.copy(features_2)
            pose_1 = np.copy(pose_2)

            cv2.imshow(self.img_obj.image_list[0].split('\\')[-2], image_2)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        cv2.destroyAllWindows()

        if bundle_adjustment_enabled:
            plot_dir = os.path.join(script_dir, 'Results with Bundle Adjustment')
        else:
            plot_dir = os.path.join(script_dir, 'Results')

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.xlabel('Image Index')
        plt.ylabel('Reprojection Error')
        plt.title('Reprojection Error Plot')
        plt.savefig(os.path.join(plot_dir, 'reprojection_errors.png'))
        plt.close()

        if total_points.size > 0 and total_colors.size > 0:
            if total_points.shape[0] > 1:
                total_points = total_points[1:]
                total_colors = total_colors[1:]

            if total_points.shape[0] != total_colors.shape[0]:
                print("Warning: Points and colors count mismatch. Truncating to shorter length.")
                min_len = min(total_points.shape[0], total_colors.shape[0])
                total_points = total_points[:min_len]
                total_colors = total_colors[:min_len]
        else:
            print("Error: No points or colors to save. Skipping point cloud generation.")
            return

        print(f"Total points to save: {total_points.shape[0]}")
        print(f"Total colors to save: {total_colors.shape[0]}")

        self.save_to_ply(
            script_dir,  # Changed from self.img_obj.path
            total_points,
            total_colors,
            bundle_adjustment_enabled,
            binary_format=True,
            scaling_factor=self.params['scaling_factor']
        )
        print("Saved the results to point_cloud.ply file!!!")

if __name__ == '__main__':
    sfm = StructurefromMotion(r"Code/templeR")
    # sfm()
    sfm(bundle_adjustment_enabled=True) 


