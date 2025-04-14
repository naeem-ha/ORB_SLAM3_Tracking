import cv2
import numpy as np
import matplotlib.pyplot as plt
from python_orb_slam3 import ORBExtractor


import numpy as np
import cv2

class ORBMatcher3D:
    def __init__(self, camera_matrix, scale_factors, th_low=50, nn_ratio=0.75,
                 max_radius=5.0, histo_length=30, check_orientation=True):
        self.K = camera_matrix
        self.scale_factors = scale_factors
        self.th_low = th_low
        self.nn_ratio = nn_ratio
        self.max_radius = max_radius
        self.histo_length = histo_length
        self.check_orientation = check_orientation

    def project_point(self, Rcw, tcw, point_3d):
        """ Project a 3D point into the current frame """
        Xc = (Rcw @ point_3d.reshape(3, 1) + tcw).flatten()  # Ensure shape (3,)
        if Xc[2] <= 0:
            return None
        uv = self.K @ Xc
        return uv[:2] / Xc[2]


    def predict_scale(self, dist):
        """ Predict scale level based on distance and scale factors """
        for i, sf in enumerate(self.scale_factors):
            if dist < sf:
                return i
        return len(self.scale_factors) - 1

    def get_features_in_radius(self, keypoints, proj, radius, level_min, level_max):
        """ Find keypoints in radius around projection """
        x, y = proj
        nearby = []
        for i, kp in enumerate(keypoints):
            if level_min <= kp.octave <= level_max:
                dist = np.hypot(kp.pt[0] - x, kp.pt[1] - y)
                if dist <= radius:
                    nearby.append(i)
        return nearby

    def compute_orientation_bins(self, matches, kp1, kp2):
        """ Orientation histogram (30 bins) """
        bins = [[] for _ in range(self.histo_length)]
        factor = 360.0 / self.histo_length
        for m in matches:
            angle_diff = kp1[m.queryIdx].angle - kp2[m.trainIdx].angle
            if angle_diff < 0:
                angle_diff += 360
            bin_idx = int(round(angle_diff / factor)) % self.histo_length
            bins[bin_idx].append(m)
        return bins

    def top_orientation_bins(self, hist):
        lengths = [len(b) for b in hist]
        return set(np.argsort(lengths)[-3:])

    def match(self, map_points, keypoints, descriptors, Rcw, tcw):
        """
        Arguments:
            map_points: list of dicts with keys:
                'pos': np.array([x, y, z])
                'desc': 1x32 ORB descriptor
                'normal': np.array([nx, ny, nz])
                'angle': float (optional, for orientation check)
            keypoints: list of cv2.KeyPoint
            descriptors: np.array of shape (N, 32)
            Rcw: Rotation matrix (3x3)
            tcw: Translation vector (3x1)

        Returns:
            matches: list of cv2.DMatch
        """
        matches = []
        rot_hist = [[] for _ in range(self.histo_length)]

        for mp_idx, mp in enumerate(map_points):
            p3d = mp['pos']
            desc = mp['desc']
            normal = mp['normal']

            proj = self.project_point(Rcw, tcw, p3d)
            if proj is None:
                continue

            p3d = p3d.reshape(3, 1)
            PO = Rcw.T @ (p3d - tcw)
            dist = np.linalg.norm(PO)

            if np.dot(PO.flatten() / dist, normal) < 0.5:

                continue

            level = self.predict_scale(dist)
            radius = self.max_radius * self.scale_factors[level]
            indices = self.get_features_in_radius(keypoints, proj, radius, level - 1, level + 1)

            if not indices:
                continue

            best_dist = 256
            second_best = 256
            best_idx = -1

            for i in indices:
                d = descriptors[i]
                dist_ham = cv2.norm(desc, d, cv2.NORM_HAMMING)
                if dist_ham < best_dist:
                    second_best = best_dist
                    best_dist = dist_ham
                    best_idx = i
                elif dist_ham < second_best:
                    second_best = dist_ham

            if best_dist <= self.th_low and best_dist < self.nn_ratio * second_best:
                match = cv2.DMatch(_queryIdx=mp_idx, _trainIdx=best_idx, _distance=best_dist)
                matches.append(match)

                if self.check_orientation:
                    mp_angle = mp.get('angle', 0.0)
                    angle_diff = (mp_angle - keypoints[best_idx].angle + 360) % 360
                    bin_idx = int(angle_diff / (360.0 / self.histo_length))
                    rot_hist[bin_idx].append(match)

        if self.check_orientation:
            top_bins = self.top_orientation_bins(rot_hist)
            matches = [m for i, b in enumerate(rot_hist) if i in top_bins for m in b]

        return matches













# Load the images
source = cv2.imread("/home/hamza-naeem/Documents/ORB_SLAM3_Tracking/C/data/1403636579813555456.png", cv2.IMREAD_GRAYSCALE)
target = cv2.imread("/home/hamza-naeem/Documents/ORB_SLAM3_Tracking/C/data/1403636579813555456.png", cv2.IMREAD_GRAYSCALE)

# --- Method 1: OpenCV ORB ---
orb_opencv = orb_opencv = cv2.ORB_create(
    nfeatures=1000,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    firstLevel=0,
    WTA_K=2,
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=20
)
kp1_cv, des1_cv = orb_opencv.detectAndCompute(source, None)
kp2_cv, des2_cv = orb_opencv.detectAndCompute(target, None)

bf_cv = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_cv = bf_cv.match(des1_cv, des2_cv)
matches_cv = sorted(matches_cv, key=lambda x: x.distance)

matched_img_cv = cv2.drawMatches(source, kp1_cv, target, kp2_cv, matches_cv[:50], None, flags=2)

# --- Method 2: ORB-SLAM3 ORBExtractor ---
orb_slam3 = ORBExtractor()
kp1_s3, des1_s3 = orb_slam3.detectAndCompute(source)
kp2_s3, des2_s3 = orb_slam3.detectAndCompute(target)

# Check if keypoints are converted properly
if isinstance(kp1_s3[0], tuple):  # If ORBExtractor returns (x, y), convert to cv2.KeyPoint
    kp1_s3 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=20) for pt in kp1_s3]
    kp2_s3 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=20) for pt in kp2_s3]

bf_s3 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_s3 = bf_s3.match(des1_s3, des2_s3)
matches_s3 = sorted(matches_s3, key=lambda x: x.distance)

matched_img_s3 = cv2.drawMatches(source, kp1_s3, target, kp2_s3, matches_s3[:50], None, flags=2)



cv2.imwrite('OpenCV ORB Matching.jpg', matched_img_cv)


cv2.imwrite('ORB-SLAM3 ORBExtractor Matching.jpg', matched_img_s3)

K = np.array([
    [458.654, 0, 367.215],
    [0, 457.296, 248.375],
    [0, 0, 1]
])


map_points = []
for i, kp in enumerate(kp1_s3):
    if des1_s3 is None or i >= len(des1_s3):
        continue
    x, y = kp.pt
    z = 1.0  # fake depth assumption
    # Backproject to 3D using K
    X = np.linalg.inv(K) @ np.array([x * z, y * z, z])
    mp = {
        'pos': X,
        'desc': des1_s3[i],
        'normal': np.array([0, 0, 1]),  # assume facing forward
        'angle': kp.angle
    }
    map_points.append(mp)

frame_kps = kp2_s3
frame_descs = des2_s3
Rcw = np.eye(3)
tcw = np.zeros((3, 1))


matcher = ORBMatcher3D(
    camera_matrix=K,
    scale_factors=[1.0 * (1.2 ** i) for i in range(8)],
    check_orientation=True
)

matches = matcher.match(
    map_points=map_points,            # list of {pos, desc, normal, angle}
    keypoints=frame_kps,              # list of cv2.KeyPoint
    descriptors=frame_descs,          # np.ndarray Nx32
    Rcw=Rcw,                          # 3x3 np.ndarray
    tcw=tcw                           # 3x1 np.ndarray
)


# matches_s3_custom = orbslam3_matcher(des1_s3, des2_s3, kp1_s3, kp2_s3, ratio_thresh=0.75, orientation_check=True)
matched_img = cv2.drawMatches(source, kp1_s3, target, frame_kps, matches[:50], None)
cv2.imwrite('ORB-SLAM3 Style Matcher.jpg', matched_img)


#plt.tight_layout()
#plt.show()
