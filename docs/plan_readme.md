# Plan for Implementing Visual Odometry for Primary Target

## 1. Overview

The goal is to extend the current 2D object tracking system to perform visual odometry (VO) focused on the primary tracked object. This will enable the estimation of the camera's 3D motion and, subsequently, the 3D trajectory of the primary target in the camera's coordinate frame or a world frame.

This plan assumes a monocular camera setup, using `left_view.mp4`.

## 2. Core Modules to Develop/Integrate

### 2.1. Camera Calibration Module
   - **Purpose:** To obtain intrinsic camera parameters (focal length, principal point, distortion coefficients). This is crucial for accurate 3D reconstruction.
   - **Implementation:**
      - Create a script/utility for camera calibration using a chessboard or ChArUco board pattern.
      - Store calibration parameters (camera matrix `K` and distortion coefficients `D`) in a configuration file (e.g., `camera_params.yaml`).
      - Load these parameters at the start of the main application.
   - **Files:**
      - `calibrate_camera.py` (new)
      - `camera_params.yaml` (new, output of calibration)
      - `utils.py` (modify to load/store calibration)

### 2.2. Feature Detection and Matching Module (for VO)
   - **Purpose:** To detect and match salient keypoints between consecutive frames, specifically on or around the primary target.
   - **Implementation:**
      - Choose a feature detector and descriptor (e.g., ORB, SIFT, AKAZE, or potentially leverage features from a learned model if suitable for geometric tasks). ORB is often a good balance of speed and performance.
      - Detect keypoints in the current frame within or near the bounding box of the primary target.
      - Match these keypoints with keypoints detected in the previous frame (or a keyframe).
      - Use techniques like Lowe's ratio test and RANSAC (with Fundamental Matrix estimation) to filter outliers.
   - **Files:**
      - `vo_feature_handler.py` (new)
      - Modify `HybridTracker` or `main.py` to call this module.

### 2.3. Camera Motion Estimation (Ego-Motion) Module
   - **Purpose:** To estimate the camera's rotation (R) and translation (t) between consecutive frames (or keyframes) using the matched 2D keypoints.
   - **Implementation:**
      - Use the camera intrinsics (`K`) and the matched 2D keypoints.
      - Estimate the Essential Matrix (`E`) from the keypoint matches (e.g., using the 5-point or 8-point algorithm with RANSAC).
      - Decompose `E` to get `R` and `t`. Handle chirality check to select the correct pose.
      - For monocular VO, `t` will be up to a scale factor.
   - **Files:**
      - `motion_estimator.py` (new)
      - Integrate with `vo_feature_handler.py`.

### 2.4. 3D Point Triangulation Module (for Target)
   - **Purpose:** To estimate the 3D coordinates of the matched keypoints on the primary target.
   - **Implementation:**
      - Use the estimated camera poses (`R`, `t`) for two views (e.g., current and previous frame/keyframe) and the corresponding 2D matched keypoints.
      - Perform triangulation (e.g., Direct Linear Transform - DLT).
      - These 3D points will be relative to the camera frame at the first view used in triangulation.
   - **Files:**
      - `triangulator.py` (new)
      - Integrate with `motion_estimator.py`.

### 2.5. Trajectory Management and Visualization Module
   - **Purpose:** To accumulate the camera poses to form a trajectory and to estimate the 3D trajectory of the primary target.
   - **Implementation:**
      - Store the sequence of estimated camera poses (`R_world`, `t_world`) relative to an initial world frame (e.g., the first camera pose is identity).
      - For the primary target:
         - Identify a stable set of 3D points on the target (or its centroid).
         - Transform these target points into the world frame using the current camera pose.
      - Visualize the 3D camera trajectory and the 3D target trajectory (e.g., using OpenCV's Viz module, Matplotlib 3D, or a dedicated 3D library if more advanced visualization is needed).
   - **Files:**
      - `trajectory_manager.py` (new)
      - Modify `main.py` for visualization.

## 3. Integration with Existing Codebase

### 3.1. `HybridTracker` Modifications:
   - The `HybridTracker` identifies the primary target and its 2D bounding box. This information will be crucial.
   - Pass the current frame and the primary target's bounding box to the `vo_feature_handler.py`.
   - Potentially store keypoints associated with the primary target's ID if they need to be tracked over longer periods for VO purposes (distinct from Re-ID features).

### 3.2. `main.py` Modifications:
   - Initialize camera calibration and load parameters.
   - Instantiate VO-related modules.
   - In the main loop, after object tracking:
      - If a primary target is identified:
         - Undistort the current frame using camera calibration data.
         - Pass the undistorted frame and target information to the VO pipeline.
         - Get the updated camera pose and target 3D position.
         - Update trajectories.
         - Visualize 2D tracking as before, and add 3D trajectory visualization.
   - Handle initialization of the VO pipeline (e.g., the first frame or when the primary target is first detected).

## 4. Detailed Steps and Considerations:

### Step 1: Camera Calibration (Offline)
   - Implement `calibrate_camera.py`.
   - Capture calibration images/video.
   - Run calibration and save `camera_params.yaml`.

### Step 2: Basic VO Pipeline Setup (Monocular)
   - Implement `vo_feature_handler.py`:
      - Keypoint detection (e.g., ORB) within the primary target's ROI.
      - Keypoint matching (e.g., BFMatcher with Hamming distance for ORB).
   - Implement `motion_estimator.py`:
      - Essential Matrix estimation (cv2.findEssentialMat).
      - Pose recovery (cv2.recoverPose).
   - Integrate into `main.py`:
      - Load camera params.
      - For each frame, after primary target detection:
         - Detect & match features.
         - Estimate camera pose relative to the *previous* frame.
         - Accumulate poses to get the trajectory relative to the first frame.
         - **Initial focus:** Get the camera's 3D trajectory.

### Step 3: Target 3D Localization
   - Implement `triangulator.py`:
      - `cv2.triangulatePoints`.
   - In `motion_estimator.py` or a new module:
      - Select a few stable, matched keypoints *on the primary target*.
      - Triangulate their 3D positions.
      - The 3D position of the target could be the centroid of these triangulated points.
      - Transform this 3D position into the world coordinate system.

### Step 4: Refinements and Robustness
   - **Keyframe Strategy:** To reduce drift and computational load, implement a keyframe-based VO. Process every frame for tracking, but only update VO pose estimation when the camera has moved significantly or enough time has passed. Triangulate points against the last keyframe.
   - **Scale Ambiguity (Monocular):**
      - For monocular VO, the translation `t` is unitless. The trajectory will be correct in shape but not in absolute scale.
      - If the target has a known size, or if there's an object of known size in the scene, this could be used to estimate the scale.
   - **Loop Closure and Bundle Adjustment (Advanced):** For longer trajectories, consider implementing loop closure detection and bundle adjustment to correct accumulated drift. This is a more advanced topic.
   - **Handling Lost Tracks:** If the primary target is lost by the 2D tracker, the VO for the target needs to pause or re-initialize when the target is re-acquired.

## 5. Potential Challenges
   - **Computational Load:** Adding VO will increase processing time. Optimization (e.g., parameter tuning, efficient feature choices, GPU acceleration where possible for VO parts) will be needed.
   - **Drift:** Monocular VO is prone to drift over time. Keyframing and potentially loop closure are needed for longer sequences.
   - **Feature Quality:** The quality and stability of features on the target object will significantly impact VO accuracy. The target might be textureless or change appearance.
   - **Dynamic Scenes:** If other objects move significantly relative to the primary target and the background, they might confuse feature matching if not handled carefully (e.g., by focusing features on the target).

## 6. Evaluation
   - **Qualitative:** Visual inspection of the plotted 3D camera and target trajectories.
   - **Quantitative (if ground truth is available):** Compare against ground truth trajectories (e.g., from a motion capture system or simulation). For datasets like KITTI, there are established metrics.

## 7. Code Structure (New/Modified Files Summary)

- **New Files:**
    - `plan_readme.md` (this file)
    - `calibrate_camera.py`
    - `camera_params.yaml` (generated)
    - `vo_feature_handler.py`
    - `motion_estimator.py`
    - `triangulator.py`
    - `trajectory_manager.py`
- **Modified Files:**
    - `main.py` (significant changes for VO integration, visualization)
    - `hybrid_tracker.py` (to provide target info to VO pipeline)
    - `utils.py` (for camera calibration data handling)
    - `README.md` (to document the new VO feature)

This plan provides a structured approach to implementing monocular visual odometry. Start with the basics (camera calibration, monocular VO for camera trajectory) and incrementally add features like target localization and robustness improvements.
