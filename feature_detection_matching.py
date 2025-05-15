import cv2
import numpy as np

def feature_matching(self, image_0, image_1) -> tuple:
    sift = cv2.SIFT_create(
        nfeatures=self.params['sift_features'],
        contrastThreshold=self.params['sift_contrast']
    )
    gray0 = cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    key_points0, descriptors_0 = sift.detectAndCompute(gray0, None)
    key_points1, descriptors_1 = sift.detectAndCompute(gray1, None)

    if descriptors_0 is None or descriptors_1 is None:
        print("Warning: No features found in one or both images")
        return np.array([]), np.array([])

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors_0, descriptors_1, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < self.params['ratio_thresh'] * n.distance:
            good_matches.append(m)

    if len(good_matches) < self.params['min_matches']:
        print(f"Warning: Only found {len(good_matches)} matches")
        return np.array([]), np.array([])

    pts0 = np.float32([key_points0[m.queryIdx].pt for m in good_matches])
    pts1 = np.float32([key_points1[m.trainIdx].pt for m in good_matches])

    if len(pts0) > 8:
        F, mask = cv2.findFundamentalMat(
            pts0, pts1,
            cv2.FM_RANSAC,
            ransacReprojThreshold=3.0,
            confidence=0.99,
            maxIters=5000
        )
        if mask is not None:
            pts0 = pts0[mask.ravel() == 1]
            pts1 = pts1[mask.ravel() == 1]

    print(f"Found {len(pts0)} good matches after filtering")
    return pts0, pts1