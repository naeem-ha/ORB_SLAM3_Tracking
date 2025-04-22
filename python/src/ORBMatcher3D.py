import cv2
import numpy as np
from typing import List, Dict, Optional

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
        self.th_high = 100  # From ORBMatcher
        self.mb = 0.5  # Placeholder baseline for scale prediction

    def project_point(self, Rcw, tcw, point_3d):
        """Project a 3D point into the current frame."""
        Xc = (Rcw @ point_3d.reshape(3, 1) + tcw).flatten()
        if Xc[2] <= 0:
            return None
        uv = self.K @ Xc
        return uv[:2] / Xc[2]

    def predict_scale(self, dist):
        """Predict scale level based on distance."""
        log_scale = np.log(dist / self.mb)
        level = int(np.round(log_scale / np.log(self.scale_factors[1])))
        return max(0, min(len(self.scale_factors) - 1, level))

    def get_features_in_radius(self, keypoints, proj, radius, level_min, level_max):
        """Find keypoints in radius around projection."""
        if proj is None:
            return []
        x, y = proj
        nearby = []
        for i, kp in enumerate(keypoints):
            if level_min != -1 and kp.octave < level_min:
                continue
            if level_max != -1 and kp.octave > level_max:
                continue
            dist = np.hypot(kp.pt[0] - x, kp.pt[1] - y)
            if dist <= radius:
                nearby.append(i)
        return nearby

    def radius_by_viewing_cos(self, viewCos: float) -> float:
        """Determine search radius based on viewing cosine."""
        return 2.5 if viewCos > 0.998 else 4.0

    def compute_orientation_bins(self, matches, kp1_angles, keypoints):
        """Orientation histogram (histo_length bins)."""
        bins = [[] for _ in range(self.histo_length)]
        factor = 360.0 / self.histo_length
        for m in matches:
            angle_diff = (kp1_angles[m.queryIdx] - keypoints[m.trainIdx].angle + 360) % 360
            bin_idx = int(round(angle_diff / factor)) % self.histo_length
            bins[bin_idx].append(m)
        return bins

    def top_orientation_bins(self, hist):
        """Find indices of the top three histogram bins."""
        max1, max2, max3 = 0, 0, 0
        ind1, ind2, ind3 = -1, -1, -1

        for i in range(self.histo_length):
            s = len(hist[i])
            if s > max1:
                max3, max2, max1 = max2, max1, s
                ind3, ind2, ind1 = ind2, ind1, i
            elif s > max2:
                max3, max2 = max2, s
                ind3, ind2 = ind2, i
            elif s > max3:
                max3 = s
                ind3 = i

        if max2 < 0.1 * max1:
            ind2 = -1
            ind3 = -1
        elif max3 < 0.1 * max1:
            ind3 = -1

        return {ind1, ind2, ind3} - {-1}

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
        kp1_angles = [mp.get('angle', 0.0) for mp in map_points]

        # Create transformation matrix
        Tcw = np.eye(4)
        Tcw[:3, :3] = Rcw
        Tcw[:3, 3] = tcw.flatten()

        # Image bounds
        mnMinX, mnMaxX = 0, 640  # Adjust based on image size
        mnMinY, mnMaxY = 0, 480

        for mp_idx, mp in enumerate(map_points):
            p3d = mp['pos']
            desc = mp['desc']
            normal = mp['normal']

            # Project point
            Xc = Tcw[:3, :3] @ p3d + Tcw[:3, 3]
            depth = Xc[2]
            if depth <= 0:
                continue

            u = self.K[0, 0] * (Xc[0] / Xc[2]) + self.K[0, 2]
            v = self.K[1, 1] * (Xc[1] / Xc[2]) + self.K[1, 2]
            if not (mnMinX <= u <= mnMaxX and mnMinY <= v <= mnMaxY):
                continue

            # Viewing angle check
            PO = Xc / depth
            viewCos = np.dot(PO, normal)
            if viewCos < 0.5:
                continue

            # Predict scale
            nPredictedLevel = self.predict_scale(depth)
            r = self.radius_by_viewing_cos(viewCos) * self.scale_factors[nPredictedLevel]

            # Find nearby keypoints
            vIndices = self.get_features_in_radius(
                keypoints, (u, v), r,
                nPredictedLevel - 1, nPredictedLevel
            )

            if not vIndices:
                continue

            bestDist = 256
            bestIdx = -1
            bestDist2 = 256
            bestLevel = -1
            bestLevel2 = -1

            for idx in vIndices:
                d = descriptors[idx]
                dist = cv2.norm(desc, d, cv2.NORM_HAMMING)
                if dist < bestDist:
                    bestDist2 = bestDist
                    bestDist = dist
                    bestLevel2 = bestLevel
                    bestLevel = keypoints[idx].octave
                    bestIdx = idx
                elif dist < bestDist2:
                    bestLevel2 = keypoints[idx].octave
                    bestDist2 = dist

            if bestDist <= self.th_high:
                if bestLevel == bestLevel2 and bestDist > self.nn_ratio * bestDist2:
                    continue

                if bestDist <= self.th_low and bestDist < self.nn_ratio * bestDist2:
                    match = cv2.DMatch(_queryIdx=mp_idx, _trainIdx=bestIdx, _distance=float(bestDist))
                    matches.append(match)
                    if self.check_orientation:
                        rot_hist.append(match)  # Will process bins later

        if self.check_orientation:
            rot_hist = self.compute_orientation_bins(matches, kp1_angles, keypoints)
            top_bins = self.top_orientation_bins(rot_hist)
            matches = [m for i, b in enumerate(rot_hist) if i in top_bins for m in b]

        return matches