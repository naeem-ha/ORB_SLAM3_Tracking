import cv2
import numpy as np


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
