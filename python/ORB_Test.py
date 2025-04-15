import pygame
import cv2
import numpy as np
from typing import List, Tuple, Optional
from src.ORB_Extractor import ORBExtractor
from src.ORBMatcher3D import ORBMatcher3D


class Camera:
    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

class MapPoint:
    def __init__(self, position: np.ndarray, descriptor: np.ndarray, normal: np.ndarray = None):
        self.position = position
        self.descriptor = descriptor
        self.normal = normal if normal is not None else np.array([0, 0, 1], dtype=np.float32)

class Frame:
    def __init__(self, image: np.ndarray, color_image: np.ndarray, orb_extractor: 'ORBExtractor', camera: Camera):
        self.image = image
        self.color_image = color_image
        self.camera = camera
        self.keypoints, self.descriptors, _ = orb_extractor(image)
        if len(self.keypoints) != len(self.descriptors):
            print(f"Warning: keypoints ({len(self.keypoints)}) and descriptors ({len(self.descriptors)}) misaligned")
            min_len = min(len(self.keypoints), len(self.descriptors))
            self.keypoints = self.keypoints[:min_len]
            self.descriptors = self.descriptors[:min_len]
        self.pose = np.eye(4, dtype=np.float32)

class Tracking:
    def __init__(self, camera: Camera, orb_extractor: 'ORBExtractor', orb_matcher: 'ORBMatcher3D'):
        self.camera = camera
        self.orb_extractor = orb_extractor
        self.orb_matcher = orb_matcher
        self.active_map = []
        self.last_frame = None
        self.trajectory = []
        self.frame_count = 0
        self.initialized = False
        self.last_pose = np.eye(4, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.angular_velocity = np.zeros(3, dtype=np.float32)
        self.tracking_failure_count = 0
        self.matches = []
        self.raw_matches = []

    def initialize_map(self, frame1: Frame, frame2: Frame):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(frame1.descriptors, frame2.descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < 70]

        print(f"Initialization matches: {len(good_matches)}")
        if len(good_matches) < 50:
            return False

        pts1 = np.float32([frame1.keypoints[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([frame2.keypoints[m.trainIdx].pt for m in good_matches])

        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        if F is None or mask is None:
            return False

        inliers = mask.ravel().astype(bool)
        pts1 = pts1[inliers]
        pts2 = pts2[inliers]
        good_matches = [good_matches[i] for i in range(len(good_matches)) if inliers[i]]

        if len(good_matches) < 30:
            return False

        E = self.camera.K.T @ F @ self.camera.K
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.camera.K)

        T1 = np.eye(4, dtype=np.float32)
        T2 = np.eye(4, dtype=np.float32)
        T2[:3, :3] = R
        T2[:3, 3] = t.flatten() * 0.1  # Scale down translation

        projMatr1 = self.camera.K @ T1[:3]
        projMatr2 = self.camera.K @ T2[:3]
        points_4d = cv2.triangulatePoints(projMatr1, projMatr2, pts1.T, pts2.T)
        points_3d = (points_4d[:3] / points_4d[3]).T * 0.05

        for i, pt_3d in enumerate(points_3d):
            if not np.isfinite(pt_3d).all() or pt_3d[2] <= 0:
                continue
            descriptor = frame1.descriptors[good_matches[i].queryIdx]
            self.active_map.append(MapPoint(pt_3d, descriptor))

        print(f"Initialized map with {len(self.active_map)} points")
        frame1.pose = T1
        frame2.pose = T2
        self.trajectory.append(T1)
        self.trajectory.append(T2)
        self.initialized = True
        self.last_pose = T2

        self.velocity = T2[:3, 3] - T1[:3, 3]
        R1 = T1[:3, :3]
        R2 = T2[:3, :3]
        R_delta = R2 @ R1.T
        rvec, _ = cv2.Rodrigues(R_delta)
        self.angular_velocity = rvec.flatten()
        return True

    def predict_pose(self) -> np.ndarray:
        if self.last_frame is None or len(self.trajectory) < 2:
            return np.eye(4, dtype=np.float32)
        if self.last_frame is not None and self.frame_count > 2:
            prev_kps = np.float32([kp.pt for kp in self.last_frame.keypoints])
            curr_kps, status, _ = cv2.calcOpticalFlowPyrLK(
                self.last_frame.image,
                self.current_image,
                prev_kps,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            if curr_kps is not None and status is not None:
                status = status.flatten()
                good_mask = status == 1
                good_prev = prev_kps[good_mask]
                good_curr = curr_kps[good_mask]
                if len(good_prev) > 10:
                    E, mask = cv2.findEssentialMat(good_curr, good_prev, self.camera.K, method=cv2.RANSAC)
                    if E is not None:
                        _, R, t, mask = cv2.recoverPose(E, good_curr, good_prev, self.camera.K)
                        predicted_pose = np.eye(4, dtype=np.float32)
                        predicted_pose[:3, :3] = R
                        predicted_pose[:3, 3] = t.flatten() * 0.1  # Scale down translation
                        return predicted_pose

        predicted_pose = self.last_pose.copy()
        predicted_pose[:3, 3] += self.velocity * 0.5  # Scale down velocity contribution
        R = predicted_pose[:3, :3]
        rvec, _ = cv2.Rodrigues(R)
        rvec_pred = rvec + self.angular_velocity * 0.5
        R_pred, _ = cv2.Rodrigues(rvec_pred)
        predicted_pose[:3, :3] = R_pred
        return predicted_pose

    def track(self, image: np.ndarray, color_image: np.ndarray) -> Optional[np.ndarray]:
        self.current_image = image
        current_frame = Frame(image, color_image, self.orb_extractor, self.camera)
        self.frame_count += 1

        if not self.initialized:
            if self.frame_count == 1:
                self.last_frame = current_frame
                return None
            elif self.frame_count == 2:
                success = self.initialize_map(self.last_frame, current_frame)
                if not success:
                    return None
                self.last_frame = current_frame
                return current_frame.pose

        map_points = [
            {'pos': mp.position, 'desc': mp.descriptor, 'normal': mp.normal, 'angle': 0.0}
            for mp in self.active_map
        ]
        predicted_pose = self.predict_pose()
        Rcw = predicted_pose[:3, :3]
        tcw = predicted_pose[:3, 3]

        if map_points:
            visible_points = 0
            for mp in map_points:
                point = mp['pos']
                point_cam = Rcw @ point + tcw
                if point_cam[2] > 0:
                    x = self.camera.fx * point_cam[0] / point_cam[2] + self.camera.cx
                    y = self.camera.fy * point_cam[1] / point_cam[2] + self.camera.cy
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        visible_points += 1
                    else:
                        print(f"Map point projects outside frame: ({x}, {y}), 3D pos: {point}")
            print(f"Visible map points: {visible_points}/{len(map_points)}")

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        map_descriptors = np.array([mp['desc'] for mp in map_points])
        if len(map_descriptors) != len(map_points):
            print(f"Error: map_descriptors ({len(map_descriptors)}) and map_points ({len(map_points)}) misaligned")
            return None
        self.raw_matches = matcher.match(map_descriptors, current_frame.descriptors)
        self.raw_matches = sorted(self.raw_matches, key=lambda x: x.distance)
        good_raw_matches = [m for m in self.raw_matches if m.distance < 100]
        print(f"Raw matches (before projection checks): {len(good_raw_matches)}")

        matches = self.orb_matcher.match(
            map_points, current_frame.keypoints, current_frame.descriptors, Rcw, tcw
        )

        print(f"Number of keypoints in frame: {len(current_frame.keypoints)}")
        print(f"Number of map points: {len(map_points)}")
        print(f"Matches (after projection checks): {len(matches)}")
        if matches:
            query_indices = [m.queryIdx for m in matches]
            print(f"Query indices: {query_indices}")
            print(f"Max query index: {max(query_indices) if query_indices else -1}")

        if len(matches) < 5 and len(good_raw_matches) >= 5:
            print("Using raw matches for tracking due to projection check failure")
            matches = good_raw_matches

        if len(matches) < 5:
            print(f"Tracking failed: insufficient matches ({len(matches)})")
            self.tracking_failure_count += 1
            if self.tracking_failure_count > 20:
                print("Too many tracking failures, reinitializing...")
                self.initialized = False
                self.active_map = []
                self.trajectory = []
                self.matches = []
                self.raw_matches = []
                self.tracking_failure_count = 0
            self.matches = matches
            # Still append the predicted pose to keep the trajectory dense
            self.trajectory.append(predicted_pose)
            self.last_pose = predicted_pose
            return None

        self.tracking_failure_count = 0
        self.matches = matches
        kps_curr, pts_3d = self.get_matched_points(current_frame, matches)
        optimized_pose = self.optimize_pose(kps_curr, pts_3d, predicted_pose)

        current_frame.pose = optimized_pose
        self.last_frame = current_frame
        self.trajectory.append(optimized_pose)

        if len(self.trajectory) >= 2:
            self.velocity = self.trajectory[-1][:3, 3] - self.trajectory[-2][:3, 3]
            R1 = self.trajectory[-2][:3, :3]
            R2 = self.trajectory[-1][:3, :3]
            R_delta = R2 @ R1.T
            rvec, _ = cv2.Rodrigues(R_delta)
            self.angular_velocity = rvec.flatten()

        self.update_map(current_frame, matches)
        self.last_pose = optimized_pose

        print(f"Tracking successful: {len(matches)} matches, pose translation: {optimized_pose[:3, 3]}")
        return optimized_pose

    def get_matched_points(self, current_frame: Frame, matches: List[cv2.DMatch]) -> Tuple[np.ndarray, np.ndarray]:
        kps_curr = []
        pts_3d = []

        for match in matches:
            map_idx = match.queryIdx
            frame_idx = match.trainIdx
            if map_idx >= len(self.active_map):
                print(f"Invalid map_idx {map_idx}, active_map size: {len(self.active_map)}")
                continue
            if frame_idx >= len(current_frame.keypoints):
                print(f"Invalid frame_idx {frame_idx}, keypoints size: {len(current_frame.keypoints)}")
                continue
            kps_curr.append(current_frame.keypoints[frame_idx].pt)
            pts_3d.append(self.active_map[map_idx].position)

        kps_curr = np.array(kps_curr, dtype=np.float32)
        pts_3d = np.array(pts_3d, dtype=np.float32)
        return kps_curr, pts_3d

    def optimize_pose(self, kps_2d: np.ndarray, pts_3d: np.ndarray, initial_pose: np.ndarray) -> np.ndarray:
        R = initial_pose[:3, :3]
        t = initial_pose[:3, 3]
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.reshape(3, 1)
        rvec = rvec.reshape(3, 1)

        print(f"optimize_pose: kps_2d shape: {kps_2d.shape}, pts_3d shape: {pts_3d.shape}")
        print(f"rvec shape: {rvec.shape}, tvec shape: {tvec.shape}")

        success, rvec, tvec = cv2.solvePnP(
            pts_3d, kps_2d, self.camera.K, distCoeffs=None, rvec=rvec, tvec=tvec, useExtrinsicGuess=True
        )

        if not success:
            print("solvePnP failed")
            return initial_pose

        R, _ = cv2.Rodrigues(rvec)
        optimized_pose = np.eye(4, dtype=np.float32)
        optimized_pose[:3, :3] = R
        optimized_pose[:3, 3] = tvec.flatten()
        return optimized_pose

    def update_map(self, current_frame: Frame, matches: List[cv2.DMatch]):
        if self.frame_count % 3 != 0:
            return

        matched_indices = set(m.trainIdx for m in matches)
        unmatched_indices = [i for i in range(len(current_frame.keypoints)) if i not in matched_indices]

        if not unmatched_indices or not self.last_frame:
            return

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        unmatched_desc = np.array([current_frame.descriptors[i] for i in unmatched_indices])
        matches = matcher.match(self.last_frame.descriptors, unmatched_desc)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < 70]

        if len(good_matches) < 5:
            return

        pts1 = np.float32([self.last_frame.keypoints[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([current_frame.keypoints[unmatched_indices[m.trainIdx]].pt for m in good_matches])

        T1 = self.last_frame.pose
        T2 = current_frame.pose
        projMatr1 = self.camera.K @ T1[:3]
        projMatr2 = self.camera.K @ T2[:3]
        points_4d = cv2.triangulatePoints(projMatr1, projMatr2, pts1.T, pts2.T)
        points_3d = (points_4d[:3] / points_4d[3]).T * 0.05

        for i, pt_3d in enumerate(points_3d):
            if not np.isfinite(pt_3d).all() or pt_3d[2] <= 0:
                continue
            descriptor = current_frame.descriptors[unmatched_indices[good_matches[i].trainIdx]]
            self.active_map.append(MapPoint(pt_3d, descriptor))

        print(f"Added {len(good_matches)} new points to map. Total: {len(self.active_map)}")

class Camera3D:
    def __init__(self, focal_length: float = 2000.0):  # Increased focal length for less distortion
        self.focal_length = focal_length
        self.position = np.array([0, -1, 3], dtype=np.float32)  # Move closer to the trajectory
        pitch = np.array([
            [1, 0, 0],
            [0, np.cos(np.radians(-20)), -np.sin(np.radians(-20))],  # Reduced pitch angle
            [0, np.sin(np.radians(-20)), np.cos(np.radians(-20))]
        ])
        self.rotation = pitch

    def project(self, point_3d: np.ndarray) -> Tuple[int, int]:
        point_cam = point_3d - self.position
        point_cam = self.rotation @ point_cam

        if point_cam[2] <= 0:
            return None

        x = (self.focal_length * point_cam[0]) / point_cam[2] + WIDTH / 2
        y = (self.focal_length * point_cam[1]) / point_cam[2] + HEIGHT / 2
        return int(x), int(y)

class Visualizer:
    def __init__(self):
        self.camera = Camera3D(focal_length=2000.0)
        self.points = []
        self.poses = []

    def update(self, points: List[np.ndarray], poses: List[np.ndarray]):
        self.poses = poses

    def draw_frustum(self, screen, pose: np.ndarray, scale: float = 0.05) -> List[Tuple[int, int]]:  # Reduced scale
        corners = np.array([
            [ scale,  scale,  scale],
            [ scale, -scale,  scale],
            [-scale, -scale,  scale],
            [-scale,  scale,  scale],
            [ scale * 1.5,  scale * 1.5,  scale * 2],  # Adjusted frustum proportions
            [ scale * 1.5, -scale * 1.5,  scale * 2],
            [-scale * 1.5, -scale * 1.5,  scale * 2],
            [-scale * 1.5,  scale * 1.5,  scale * 2],
        ])

        R = pose[:3, :3]
        t = pose[:3, 3]
        corners_world = (R @ corners.T + t[:, None]).T

        projected = [self.camera.project(corner) for corner in corners_world]
        if any(p is None for p in projected):
            return None

        pygame.draw.polygon(screen, CYAN, projected[:4], 1)
        pygame.draw.polygon(screen, CYAN, projected[4:], 1)
        for i in range(4):
            pygame.draw.line(screen, CYAN, projected[i], projected[(i + 1) % 4], 1)
            pygame.draw.line(screen, CYAN, projected[i], projected[i + 4], 1)
            pygame.draw.line(screen, CYAN, projected[i + 4], projected[(i + 1) % 4 + 4], 1)

        return projected

    def draw(self, screen):
        gradient_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        center = (WIDTH // 2, HEIGHT // 2)
        max_radius = max(WIDTH, HEIGHT)
        for r in range(max_radius, 0, -10):
            alpha = int(50 * (1 - r / max_radius))
            color = (0, 50, 100, alpha)
            pygame.draw.circle(gradient_surface, color, center, r)
        screen.blit(gradient_surface, (0, 0))

        previous_projected = None
        for i, pose in enumerate(self.poses):
            projected = self.draw_frustum(screen, pose)
            if projected is None:
                previous_projected = None
                continue

            if previous_projected is not None:
                for j in range(4):
                    pygame.draw.line(screen, CYAN, previous_projected[j], projected[j], 1)
                    pygame.draw.line(screen, CYAN, previous_projected[j + 4], projected[j + 4], 1)

            previous_projected = projected

    def update_camera(self, mouse_dx: float, mouse_dy: float):
        yaw = mouse_dx * 0.005
        pitch = mouse_dy * 0.005
        yaw_matrix = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])
        pitch_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])
        self.camera.rotation = pitch_matrix @ yaw_matrix @ self.camera.rotation

class DebugVisualizer:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))

    def draw(self, frame: np.ndarray, keypoints: List[cv2.KeyPoint], raw_matches: List[cv2.DMatch], 
             matches: List[cv2.DMatch], tracker: 'Tracking', pose: Optional[np.ndarray]):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
        frame_surface = pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1))

        self.surface.blit(frame_surface, (0, 0))

        scale_x = self.width / frame.shape[1]
        scale_y = self.height / frame.shape[0]

        for kp in keypoints:
            x, y = int(kp.pt[0] * scale_x), int(kp.pt[1] * scale_y)
            pygame.draw.circle(self.surface, (0, 255, 0), (x, y), 3)

        for match in raw_matches:
            map_idx = match.queryIdx
            frame_idx = match.trainIdx
            if map_idx >= len(tracker.active_map):
                print(f"Skipping invalid map_idx {map_idx}, active_map size: {len(tracker.active_map)}")
                continue
            if frame_idx >= len(keypoints):
                print(f"Skipping invalid frame_idx {frame_idx}, keypoints size: {len(keypoints)}")
                continue
            kp = keypoints[frame_idx]
            x, y = int(kp.pt[0] * scale_x), int(kp.pt[1] * scale_y)
            pygame.draw.circle(self.surface, (255, 255, 0), (x, y), 5, 2)

        if pose is not None and matches and tracker.active_map:
            Rcw = pose[:3, :3]
            tcw = pose[:3, 3]
            for match in matches:
                map_idx = match.queryIdx
                frame_idx = match.trainIdx
                if map_idx >= len(tracker.active_map):
                    print(f"Skipping invalid map_idx {map_idx}, active_map size: {len(tracker.active_map)}")
                    continue
                if frame_idx >= len(keypoints):
                    print(f"Skipping invalid frame_idx {frame_idx}, keypoints size: {len(keypoints)}")
                    continue
                kp = keypoints[frame_idx]
                x, y = int(kp.pt[0] * scale_x), int(kp.pt[1] * scale_y)
                pygame.draw.circle(self.surface, (255, 0, 0), (x, y), 5, 2)

                map_point = tracker.active_map[map_idx].position
                point_cam = Rcw @ map_point + tcw
                if point_cam[2] > 0:
                    x_proj = int(tracker.camera.fx * point_cam[0] / point_cam[2] + tracker.camera.cx)
                    y_proj = int(tracker.camera.fy * point_cam[1] / point_cam[2] + tracker.camera.cy)
                    x_proj = int(x_proj * scale_x)
                    y_proj = int(y_proj * scale_y)
                    pygame.draw.line(self.surface, (0, 0, 255), (x, y), (x_proj, y_proj), 1)

        debug_array = pygame.surfarray.array3d(self.surface)
        debug_array = np.transpose(debug_array, (1, 0, 2))
        debug_array = cv2.cvtColor(debug_array, cv2.COLOR_RGB2BGR)
        cv2.imshow("ORB Features Debug", debug_array)

# Main script
pygame.init()
WIDTH, HEIGHT = 800, 600
DEBUG_WIDTH, DEBUG_HEIGHT = 640, 480

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ORB-SLAM3 Video Visualization")

debug_visualizer = DebugVisualizer(DEBUG_WIDTH, DEBUG_HEIGHT)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)

orb_extractor = ORBExtractor(nfeatures=2000, scale_factor=1.2, nlevels=8)
scale_factors = orb_extractor.mv_scale_factor
camera_matrix = np.array([
    [458.654, 0, 367.215],
    [0, 457.296, 248.375],
    [0, 0, 1]
], dtype=np.float32)
orb_matcher = ORBMatcher3D(camera_matrix=camera_matrix, scale_factors=scale_factors)
camera = Camera(fx=458.654, fy=457.296, cx=367.215, cy=248.375)
tracker = Tracking(camera, orb_extractor, orb_matcher)

visualizer = Visualizer()

video_path = "output.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Could not open video")

running = True
clock = pygame.time.Clock()
last_mouse_pos = None

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEMOTION:
            if last_mouse_pos:
                dx = event.pos[0] - last_mouse_pos[0]
                dy = event.pos[1] - last_mouse_pos[1]
                visualizer.update_camera(dx, dy)
            last_mouse_pos = event.pos

    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pose = tracker.track(frame_gray, frame)

    points = [mp.position for mp in tracker.active_map]
    visualizer.update(points, tracker.trajectory)
    visualizer.draw(screen)
    pygame.display.flip()

    if tracker.last_frame:
        debug_visualizer.draw(frame, tracker.last_frame.keypoints, tracker.raw_matches, 
                             tracker.matches, tracker, pose)

    if cv2.waitKey(1) & 0xFF == 27:
        running = False

    clock.tick(30)

cap.release()
cv2.destroyAllWindows()
pygame.quit()