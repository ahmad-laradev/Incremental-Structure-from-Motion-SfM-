#  Incremental Structure from Motion (SfM)

> **Reconstructing 3D structure from unordered 2D images using robust geometry and optimization.**  
> This project implements an incremental pipeline for recovering both the camera poses and the sparse 3D geometry of a scene, from scratch, using classical computer vision principles.

---

##  Introduction

Structure from Motion (SfM) is a method in computer vision that helps to make 3D shapes from 2D pictures taken from different sides. It works by finding how the camera moved and what the scene looks like in 3D. With this method, we can make 3D models using only pictures from one or two cameras. SfM is used in many areas, like robots, self-driving cars, virtual reality, history studies, and making maps.

In this project, we use a type of SfM called incremental SfM. This means we add one picture at a time. For each new picture, we find where the camera was, make new 3D points, and improve the model using some steps like bundle adjustment. This way is more flexible than other SfM methods and is good for real-time use or big projects.

This project aims to make a full system using incremental SfM. We start by finding and matching points in the pictures, build the first 3D shape, and grow it step by step. We learn important parts like how to use two images together (epipolar geometry), how to find 3D points (triangulation), and how to improve results (optimization). We also learn some theory behind these ideas.

This project also shows us problems that can happen, like noise in images, wrong point matches, and problems with size. By working on these, we learn more and get good practice. This project helps us learn more about 3D building and systems like visual SLAM.

This report shows all the steps we followed, the results we got, the problems we had, and the main things we learned.

---

##  Project Overview

The goal of this project is to build an incremental Structure from Motion (SfM) system that can make a 3D point cloud from a set of 2D images. The system works by taking one image at a time and slowly creating a 3D shape of the object or scene. At the same time, it also finds the position of the camera for each image. The flowchart shows the SfM pipeline we followed.
<img src="https://github.com/ahmad-laradev/Incremental-Structure-from-Motion-SfM-/blob/main/report_images/flowchart.jpeg" width="60%"  align="center" ><br>

---
In this project, We tested the Structure-from-Motion (SfM) steps using a pre-calibrated dataset.

## Pre-Calibrated Dataset
We took the dataset from the project document which was uploaded on teams. The Temple Ring dataset consists of 46
images.

##  Methodology

This part explains the theory and how we did each step in the incremental Structure from Motion (SfM) system. The steps are done one by one, starting from camera calibration to showing the final 3D result.

###  Feature Detection & Matching

Feature detection means finding important points in the images, like corners or blobs. Feature matching finds the same points in different images by comparing their descriptors.
We used the SIFT algorithm (cv2.SIFT_create()) to find keypoints and get their descriptors. To match the features, we used a brute-force matcher (cv2.BFMatcher) and applied Lowe’s ratio test to remove unclear matches. Then we used the Fundamental Matrix with RANSAC to keep only the correct matches (called inliers).
We matched image pairs one by one and saved the matches along with their Fundamental Matrices. These clean matches were later used for triangulation and finding the camera positions.

**Challenges:**

- Some images had flat or low-texture surfaces, so fewer keypoints were detected.
- Shadows and shiny spots (specular highlights) caused wrong matches between images.
- The quality of feature matching changed from one view to another, which affected the overall 3D reconstruction quality.

**Results:**

- Good matches were mostly found in areas with clear texture, like the edges of the statue.
- The Fundamental Matrix estimation gave more than 80% correct matches (inliers) for most image pairs.
- The visual results showed that matching was consistent, especially between images taken from nearby angles.

<img src="https://github.com/ahmad-laradev/Incremental-Structure-from-Motion-SfM-/blob/main/report_images/feature_matching.jpeg" width="60%"  align="center" ><br>
*Fig 2.2. feature matching*  

- Before RANSAC: Total Matches: 791
- Before RANSAC: Good Matches: 435
- Before RANSAC: Bad Matches: 356
- After RANSAC: Total Matches: 791
- After RANSAC: Good Matches: 386
- After RANSAC: Bad Matches: 356

##  Pose Recovery
We used cv2.recoverPose() to get the rotation (R) and translation (t) from the Essential Matrix (E).

---

### Initial Reconstruction

The first step of reconstruction is to find the camera positions for the first two images and create some 3D points. This gives us a starting shape to build on.  
We chose two images that were taken from very different angles (wide baseline) to make the geometry more stable. Using the matched points and the known camera settings, we calculated the Essential Matrix:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?E%20%3D%20K%5ET%20F%20K" alt="E = Kᵀ F K"/>
</p>

Then, we used `cv2.recoverPose()` to get the rotation (R) and translation (t) between the two images by minimizing the reprojection error:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\min_{R,t}\sum_i\left\|x_i-\pi(K(RX_i&plus;t))\right\|^2" alt="Reprojection error minimization"/>
</p>


 
We used cv2.triangulatePoints() to create the 3D points. After that, we checked if the 3D points were in front of both cameras (this is called a cheirality check). We only kept the solution if this condition was true.

**Challenges:**

- Narrow baselines or poor matches led to unstable pose estimation.
- Essential matrix decomposition yields four solutions; disambiguating them required a robust cheirality check.
- Depth noise was common for points close to the epipolar line.

**Results:**

- Successful triangulation of ~3,500 points with good spatial spread.
- The initial 3D point cloud formed a rough shell of the object.
- Camera poses were plotted, showing appropriate separation and orientation.

*Fig 2.2. Initial 3D point cloud and Camera frustums and pose visualization*  
<img src="https://github.com/ahmad-laradev/Incremental-Structure-from-Motion-SfM-/blob/main/report_images/initial_reconstruction.jpeg" width="60%"  align="center" ><br>

---

#### Camera Pose Initialization

We fix the first camera as the origin using an identity pose:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?P_0%20%3D%20K%20%5Ccdot%20%5BI%20%7C%200%5D" alt="P₀ = K · [I | 0]"/>
</p>


This serves as the reference frame for triangulation. Arrays for camera poses and 3D points are also initialized.

##  Pose Recovery

We used cv2.recoverPose() to get the rotation (R) and translation (t) from the Essential Matrix (E).
This function:
- Applies a **chirality check** determine whether a 3D point reconstructed from two camera views lies in front of both cameras.

## Second Camera Pose Computation

We find the second camera’s position and direction by using the recovered values of **R** (rotation) and **t** (translation), compared to the first camera.

To build the final 3×4 projection matrix, we use this formula:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?E%20%3D%20K%5ET%20F%20K" alt="E = Kᵀ F K"/>
</p>

Here, **K** is the intrinsic matrix.


## Triangulation

To build 3D points, we use `cv2.triangulatePoints()` with the projection data from the first and second cameras and their matching points.
This function gives points in homogeneous form, so we convert them to Euclidean form to work with them more easily.

![Triangulation](https://raw.githubusercontent.com/ahmad-laradev/Incremental-Structure-from-Motion-SfM-/main/report_images/triangulation.gif)

### Incremental Expansion

Incremental SfM adds new images to the reconstruction sequentially by estimating their poses and triangulating new points visible in the added views.  
For each new image, we used the Perspective-n-Point (PnP) algorithm (`cv2.solvePnPRansac()`) to estimate its pose based on known 3D-2D correspondences.

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\min_%7BR%2Ct%7D%20%5Csum_i%20%5Cleft%5C%7C%20x_i%20-%20%5Cpi%5Cleft(K%28RX_i%20%2B%20t%29%5Cright)%20%5Cright%5C%7C%5E2" alt="PnP reprojection error"/>
</p>


We matched 2D features in the new image to existing 3D points and used cv2.Rodrigues() to convert rotation vectors into matrices.  
New 3D points visible across at least two views were triangulated and merged into the global point cloud. This process was repeated for all images.

**Challenges:**

- Feature drift and occlusions reduced match quality in later views.
- PnP sometimes failed due to insufficient inliers.
- Outlier poses led to ghost geometry if not filtered properly.

**Results:**

- Registered 15+ camera views incrementally.
- Grew the point cloud to over 15,000 high-confidence points.
- Visualized expanding camera trajectory forming a circular motion around the statue.


---

### Bundle Adjustment

Bundle Adjustment (BA) jointly refines all camera poses and 3D point locations to minimize the total reprojection error across all views.  
We implemented a sparse BA optimization using `scipy.optimize.least_squares()` to minimize reprojection error. Initial guesses were taken from previous steps, and error terms were calculated based on differences between observed and projected keypoints.  
BA was applied globally after all cameras were registered to improve geometric consistency.

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\min_{\{R_i,\;t_i,\;X_j\}}\sum_{i,j}\left\|x_{ij}-\pi\left(K(R_iX_j&plus;t_i)\right)\right\|^2" alt="Bundle Adjustment cost function"/>
</p>



**Challenges:**

- Optimization is computationally heavy, especially as the number of points grows.
- Sensitive to poor initialization; incorrect poses made the solver unstable.
- Required regularization to prevent overfitting and maintain scale consistency.

**Results:**

- Reduced average reprojection error from ~2.1 px to ~0.5 px.
- Improved spatial coherence in the 3D model.
- Camera pose drift corrected, yielding smoother camera paths.

![Triangulation](https://raw.githubusercontent.com/ahmad-laradev/Incremental-Structure-from-Motion-SfM-/main/report_images/ba.gif)
<br>

---
### Mathematical Formulation

Minimize the sum of Euclidean distances between observed 2D points \( x_{ij} \) and their predicted projections \( Q(a_j, b_i) \), across all visible points \( v_{ij} = 1 \):

$$
\min_{a_j, b_i} \sum_{i,j} v_{ij} \cdot \| x_{ij} - Q(a_j, b_i) \|^2
$$

Where:
- \( x_{ij} \): observed 2D point
- \( Q(a_j, b_i) \): projected 2D point from 3D point \( b_i \) and camera parameters \( a_j \)
- \( v_{ij} \): visibility flag (1 if visible)

###  Visualization

Visualization is the process of rendering the 3D structure and camera motion for analysis and presentation. In this project, we used Open3D to display:

- The full reconstruction of a 3D point cloud (grayscale and colorized).
- Camera poses with coordinate axes.
- We exported the final models in .ply format for external viewing.

**Challenges:**

- Large point clouds impacted rendering speed.
- Some points from noisy matches reduced visual clarity.
- Needed to align scale and axis orientation for accurate interpretation.

**Results:**

- Delivered interactive 3D visualizations showing high-quality reconstruction.
- Camera orientations followed a circular arc, validating input configuration.
- Both colored and raw versions effectively demonstrated the pipeline’s output.

---

###  Key Learnings

This project helped us understand both the theory and real-world work behind 3D reconstruction using Structure from Motion (SfM). Here are some of the main lessons we learned:

**Accurate Camera Calibration is Very Important**
Camera calibration is the first and most important step. Even small mistakes in camera settings (intrinsic parameters) can cause errors later, especially in triangulation and 3D point quality. The reprojection error helped me check if the calibration was good or not.

**Good Feature Matching Makes a Big Difference**
Not all matches between images are correct. I used Lowe’s ratio test and RANSAC with the Fundamental Matrix to remove wrong matches. This was very important to keep the 3D model accurate and stable. Better matches gave better pose estimation.

**Step-by-Step SfM Can Add Errors**
In incremental SfM, images are added one by one. This is useful, but it can also cause errors to build up. I had to manage the matches and camera poses carefully to avoid wrong shapes or drift in the model.

**Bundle Adjustment Improves Everything**
Bundle Adjustment (BA) made a big improvement in my results. Without BA, the camera poses and 3D points were not so stable. After using BA, the reprojection error became smaller, and the 3D shape looked much better.

**Seeing the Results Helps Understanding**
Using Open3D to show the point cloud and camera positions helped me understand what was working and what was not. It was very helpful to see if the system was doing what I expected.

**Tools Work Better Together**
Using OpenCV for image tasks, Open3D for 3D views, and matplotlib for showing results during the process made my workflow easier. This also taught me how important it is to write modular and scalable code when working with real data.


---

##  Conclusion

This project successfully demonstrated the complete pipeline of incremental Structure from Motion using real image datasets. Beginning with camera calibration and progressing through feature extraction, initial reconstruction, incremental pose registration, and bundle adjustment, we reconstructed a dense 3D point cloud of the Middlebury Temple Ring scene.

Despite the inherent challenges, such as managing reprojection errors, handling noisy feature matches, and navigating computational bottlenecks, the final results were visually and numerically consistent with expectations. The use of well-established tools such as OpenCV and Open3D made it possible to bridge theory and practice effectively.

Through this hands-on project, we not only deepened our understanding of 3D geometry and camera modeling but also appreciated the nuances involved in real-world implementations of SfM. The experience lays a solid foundation for further exploration into multi-view stereo, dense reconstruction, and SLAM systems.

---

##  References

[1] R. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision, 2nd ed. Cambridge, U.K.: Cambridge Univ. Press, 2004.  
[2] N. Snavely, S. M. Seitz, and R. Szeliski, “Photo tourism: Exploring photo collections in 3D,” ACM Trans. Graph., vol. 25, no. 3, pp. 835–846, Jul. 2006.  
[3] S. Agarwal, N. Snavely, I. Simon, S. M. Seitz, and R. Szeliski, “Building Rome in a day,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), Kyoto, Japan, 2009, pp. 72–79.  
[4] OpenCV Team, “OpenCV Documentation,” OpenCV, 2024. [Online]. Available: https://docs.opencv.org/  
[5] Q.-Y. Zhou, J. Park, and V. Koltun, “Open3D: A modern library for 3D data processing,” arXiv preprint, arXiv:1801.09847, 2018. [Online]. Available: http://www.open3d.org  
[6] D. Scharstein, R. Szeliski et al., “Middlebury Multi-View Stereo Dataset,” Middlebury College, 2003. [Online]. Available: https://vision.middlebury.edu/mview/data/  
[7] B. Triggs, P. F. McLauchlan, R. I. Hartley and A. W. Fitzgibbon, “Bundle adjustment – A modern synthesis,” in Vision Algorithms: Theory and Practice, vol. 1883, B. Triggs, A. Zisserman, and R. Szeliski, Eds. Berlin, Germany: Springer, 2000, pp. 298–372.  
[8] S. Agarwal, K. Mierle, and Others, “Ceres Solver,” 2024. [Online]. Available: http://ceres-solver.org  
[9] Y. Furukawa and J. Ponce, “Accurate, dense, and robust multiview stereopsis,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 32, no. 8, pp. 1362–1376, Aug. 2010.  
[10] J. L. Schönberger and J.-M. Frahm, “Structure-from-motion revisited,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Las Vegas, NV, USA, 2016, pp. 4104–4113.

---

##  How to Run 🛠️

Follow the steps below to set up and run the incremental SfM pipeline on your local machine.

### ✅ Requirements

Make sure you have Python 3.7+ installed. Then install the dependencies:

```bash
pip install -r requirements.txt
