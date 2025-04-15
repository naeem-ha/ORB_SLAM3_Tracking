import cv2
import numpy as np
from typing import Tuple, List, Optional

class Camera:
    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        """Initialize camera parameters for a pin-hole model."""
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

class MapPoint:
    def __init__(self, position: np.ndarray, descriptor: np.ndarray, normal: np.ndarray = None):
        """Initialize a 3D map point with its position, ORB descriptor, and normal."""
        self.position = position  # 3D position in world coordinates
        self.descriptor = descriptor  # ORB descriptor for matching
        self.normal = normal if normal is not None else np.array([0, 0, 1], dtype=np.float32)  # Default normal

class Frame:
    def __init__(self, image: np.ndarray, orb_extractor: 'ORBExtractor', camera: Camera):
        """Extract ORB features from the image and initialize frame."""
        self.image = image
        self.camera = camera
        self.keypoints, self.descriptors, _ = orb_extractor(image)
        self.pose = np.eye(4, dtype=np.float32)  # Initial pose (4x4 homogeneous transformation)

class Tracking:
    def __init__(self, camera: Camera, orb_extractor: 'ORBExtractor', orb_matcher: 'ORBMatcher3D'):
        """Initialize the tracking thread."""
        self.camera = camera
        self.orb_extractor = orb_extractor
        self.orb_matcher = orb_matcher
        self.active_map = []  # List of MapPoint objects
        self.last_frame = None  # Store the last processed frame

    def track(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Track the current frame and estimate its pose."""
        # Create a new frame
        current_frame = Frame(image, self.orb_extractor, self.camera)

        if self.last_frame is None:
            # First frame: initialize map (simplified, in practice you'd need initialization via triangulation)
            self.last_frame = current_frame
            return current_frame.pose

        # Prepare map points for matching
        map_points = [
            {
                'pos': mp.position,
                'desc': mp.descriptor,
                'normal': mp.normal,
                'angle': 0.0  # Simplified, as MapPoint doesn't store angle
            }
            for mp in self.active_map
        ]

        # Initial pose prediction (constant velocity motion model)
        predicted_pose = self.predict_pose()
        Rcw = predicted_pose[:3, :3]
        tcw = predicted_pose[:3, 3]

        # Match features between current frame and map
        matches = self.orb_matcher.match(
            map_points, current_frame.keypoints, current_frame.descriptors, Rcw, tcw
        )

        if len(matches) < 10:  # Not enough matches to estimate pose
            return None

        # Extract matched keypoints and 3D points
        kps_curr, pts_3d = self.get_matched_points(current_frame, matches)

        # Optimize pose by minimizing reprojection error
        optimized_pose = self.optimize_pose(kps_curr, pts_3d, predicted_pose)

        current_frame.pose = optimized_pose
        self.last_frame = current_frame

        # Update map with new observations (simplified)
        self.update_map(current_frame, matches)

        return optimized_pose

    def get_matched_points(self, current_frame: Frame, matches: List[cv2.DMatch]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 2D keypoints and corresponding 3D points from matches."""
        kps_curr = []
        pts_3d = []

        for match in matches:
            map_idx = match.queryIdx
            frame_idx = match.trainIdx

            kps_curr.append(current_frame.keypoints[frame_idx].pt)
            pts_3d.append(self.active_map[map_idx].position)

        kps_curr = np.array(kps_curr, dtype=np.float32)
        pts_3d = np.array(pts_3d, dtype=np.float32)
        return kps_curr, pts_3d

    def predict_pose(self) -> np.ndarray:
        """Predict the current pose based on the last frame (constant velocity model)."""
        if self.last_frame is None:
            return np.eye(4, dtype=np.float32)

        # Simplified: assume the same pose as the last frame
        return self.last_frame.pose.copy()

    def optimize_pose(self, kps_2d: np.ndarray, pts_3d: np.ndarray, initial_pose: np.ndarray) -> np.ndarray:
        """Optimize the camera pose by minimizing reprojection error."""
        # Convert initial pose to rotation vector and translation
        R = initial_pose[:3, :3]
        t = initial_pose[:3, 3]
        rvec, _ = cv2.Rodrigues(R)
        tvec = t

        # Optimize using solvePnP
        success, rvec, tvec = cv2.solvePnP(
            pts_3d, kps_2d, self.camera.K, distCoeffs=None, rvec=rvec, tvec=tvec, useExtrinsicGuess=True
        )

        if not success:
            return initial_pose

        # Convert back to 4x4 transformation matrix
        R, _ = cv2.Rodrigues(rvec)
        optimized_pose = np.eye(4, dtype=np.float32)
        optimized_pose[:3, :3] = R
        optimized_pose[:3, 3] = tvec.flatten()
        return optimized_pose

    def update_map(self, current_frame: Frame, matches: List[cv2.DMatch]):
        """Update the active map with new observations (simplified)."""
        # In a full implementation, you'd triangulate new points and add them to the map
        pass
