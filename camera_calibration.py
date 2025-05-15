import os
import cv2
import numpy as np

class ImageLoader:
    """
    Loads images from a specified directory and handles camera intrinsics downscaling.
    """
    def __init__(self, img_dir: str, downscale_factor: float):
        self.img_dir = os.path.abspath(img_dir)
        k_file = os.path.join(self.img_dir, 'K.txt')
        with open(k_file, 'r') as f:
            lines = f.read().strip().split('\n')
            matrix_values = []
            for line in lines:
                row_vals = [float(val) for val in line.strip().split()]
                matrix_values.append(row_vals)
            self.K = np.array(matrix_values, dtype=np.float32)
        print("Original Intrinsics:\n", self.K)

        self.image_list = []
        for filename in sorted(os.listdir(self.img_dir)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.image_list.append(os.path.join(self.img_dir, filename))

        self.path = os.getcwd()
        self.factor = downscale_factor
        self.downscale_instrinsics()
        print("Downscaled Intrinsics:\n", self.K)

    def downscale_image(self, image):
        new_w = int(image.shape[1] / self.factor)
        new_h = int(image.shape[0] / self.factor)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def downscale_instrinsics(self) -> None:
        self.K[0, 0] /= self.factor
        self.K[1, 1] /= self.factor
        self.K[0, 2] /= self.factor
        self.K[1, 2] /= self.factor