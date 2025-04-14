import cv2
import numpy as np
from math import atan2, cos, sin, sqrt, pi, ceil

class ORBExtractor:
    PATCH_SIZE = 31
    HALF_PATCH_SIZE = 15
    EDGE_THRESHOLD = 19
    
    # ORB pattern from C implementation (256 pairs, each with x1,y1,x2,y2)
    BIT_PATTERN_31_ = [
        8,-3, 9,5, 4,2, 7,-12, -11,9, -8,2, 7,-12, 12,-13, 2,-13, 2,12,
        1,-7, 1,6, -2,-10, -2,-4, -13,-13, -11,-8, -13,-3, -12,-9, 10,4, 11,9,
        -13,-8, -8,-9, -11,7, -9,12, 7,7, 12,6, -4,-5, -3,0, -13,2, -12,-3,
        -9,0, -7,5, 12,-6, 12,-1, -3,6, -2,12, -6,-13, -4,-8, 11,-13, 12,-8,
        4,7, 5,1, 5,-3, 10,-3, 3,-7, 6,12, -8,-7, -6,-2, -2,11, -1,-10,
        -13,12, -8,10, -7,3, -5,-3, -4,2, -3,7, -10,-12, -6,11, 5,-12, 6,-7,
        5,-6, 7,-1, 1,0, 4,-5, 9,11, 11,-13, 4,7, 4,12, 2,-1, 4,4,
        -4,-12, -2,7, -8,-5, -7,-10, 4,11, 9,12, 0,-8, 1,-13, -13,-2, -8,2,
        -3,-2, -2,3, -6,9, -4,-9, 8,12, 10,7, 0,9, 1,3, 7,-5, 11,-10,
        -13,-6, -11,0, 10,7, 12,1, -6,-3, -6,12, 10,-9, 12,-4, -13,8, -8,-12,
        -13,0, -8,-4, 3,3, 7,8, 5,7, 10,-7, -1,7, 1,-12, 3,-10, 5,6,
        2,-4, 3,-10, -13,0, -13,5, -13,-7, -12,12, -13,3, -11,8, -7,12, -4,7,
        6,-10, 12,8, -9,-1, -7,-6, -2,-5, 0,12, -12,5, -7,5, 3,-10, 8,-13,
        -7,-7, -4,5, -3,-2, -1,-7, 2,9, 5,-11, -11,-13, -5,-13, -1,6, 0,-1,
        5,-3, 5,2, -4,-13, -4,12, -9,-6, -9,6, -12,-10, -8,-4, 10,2, 12,-3,
        7,12, 12,12, -7,-13, -6,5, -4,9, -3,4, 7,-1, 12,2, -7,6, -5,1,
        -13,11, -12,5, -3,7, -2,-6, 7,-8, 12,-7, -13,-7, -11,-12, 1,-3, 12,12,
        2,-6, 3,0, -4,3, -2,-13, -1,-13, 1,9, 7,1, 8,-6, 1,-1, 3,12,
        9,1, 12,6, -1,-9, -1,3, -13,-13, -10,5, 7,7, 10,12, 12,-5, 12,9,
        6,3, 7,11, 5,-13, 6,10, 2,-12, 2,3, 3,8, 4,-6, 2,6, 12,-13,
        9,-12, 10,3, -8,4, -7,9, -11,12, -4,-6, 1,12, 2,-8, 6,-9, 7,-4,
        2,3, 3,-2, 6,3, 11,0, 3,-3, 8,-8, 7,8, 9,3, -11,-5, -6,-4,
        -10,11, -5,10, -5,-8, -3,12, -10,5, -9,0, 8,-1, 12,-6, 4,-6, 6,-11,
        -10,12, -8,7, 4,-2, 6,7, -2,0, -2,12, -5,-8, -5,2, 7,-6, 10,12,
        -9,-13, -8,-8, -5,-13, -5,-2, 8,-8, 9,-13, -9,-11, -9,0, 1,-8, 1,-2,
        7,-4, 9,1, -2,1, -1,-4, 11,-6, 12,-11, -12,-9, -6,4, 3,7, 7,12,
        5,5, 10,8, 0,-4, 2,8, -9,12, -5,-13, 0,7, 2,12, -1,2, 1,7,
        5,11, 7,-9, 3,5, 6,-8, -13,-4, -8,9, -5,9, -3,-3, -4,-7, -3,-12,
        6,5, 8,0, -7,6, -6,12, -13,6, -5,-2, 1,-10, 3,10, 4,1, 8,-4,
        -2,-2, 2,-13, 2,-12, 12,12, -2,-13, 0,-6, 4,1, 9,3, -6,-10, -3,-5,
        -3,-13, -1,1, 7,5, 12,-11, 4,-2, 5,-7, -13,9, -9,-5, 7,1, 8,6,
        7,-8, 7,6, -7,-4, -7,1, -8,11, -7,-8, -13,6, -12,-8, 2,4, 3,9,
        10,-5, 12,3, -6,-5, -6,7, 8,-3, 9,-8, 2,-12, 2,8, -11,-2, -10,3,
        -12,-13, -7,-9, -11,0, -10,-5, 5,-3, 11,8, -2,-13, -1,12, -1,-8, 0,9,
        -13,-11, -12,-5, -10,-2, -10,11, -3,9, -2,-13, 2,-3, 3,2, -9,-13, -4,0,
        -4,6, -3,-10, -4,12, -2,-7, -6,-11, -4,9, 6,-3, 6,11, -13,11, -5,5,
        11,11, 12,6, 7,-5, 12,-2, -1,12, 0,7, -4,-8, -3,-2, -7,1, -6,7,
        -13,-12, -8,-13, -7,-2, -6,-8, -8,5, -6,-9, -5,-1, -4,5, -13,7, -8,10,
        1,5, 5,-13, 1,0, 10,-13, 9,12, 10,-1, 5,-8, 10,-9, -1,11, 1,-13,
        -9,-3, -6,2, -1,-10, 1,12, -13,1, -8,-10, 8,-11, 10,-6, 2,-13, 3,-6,
        7,-13, 12,-9, -10,-10, -5,-7, -10,-8, -8,-13, 4,-6, 8,5, 3,12, 8,-13,
        -4,2, -3,-3, 5,-13, 10,-12, 4,-13, 5,-1, -9,9, -4,3, 0,3, 3,-9,
        -12,1, -6,1, 3,2, 4,-8, -10,-10, -10,9, 8,-13, 12,12, -8,-12, -6,-5,
        2,2, 3,7, 10,6, 11,-8, 6,8, 8,-12, -7,10, -6,5, -3,-9, -3,9,
        -1,-13, -1,5, -3,-7, -3,4, -8,-2, -8,3, 4,2, 12,12, 2,-5, 3,11,
        6,-9, 11,-13, 3,-1, 7,12, 11,-1, 12,4, -3,0, -3,6, 4,-11, 4,12,
        2,-4, 2,1, -10,-6, -8,1, -13,7, -11,1, -13,12, -11,-13, 6,0, 11,-13,
        0,-1, 1,4, -13,3, -9,-2, -9,8, -6,-3, -13,-6, -8,-2, 5,-9, 8,10,
        2,7, 3,-9, -1,-6, -1,-1, 9,5, 11,-2, 11,-3, 12,-8, 3,0, 3,5,
        -1,4, 0,10, 3,-6, 4,5, -13,0, -10,5, 5,8, 12,11, 8,9, 9,-6,
        7,-4, 8,-12, -10,4, -10,9, 7,3, 12,4, 9,-7, 10,-2, 7,0, 12,-2,
        -1,-6, 0,-11
    ]
    
    # Convert to 512 (x, y) pairs for Python implementation
    ORB_PATTERN = np.array(BIT_PATTERN_31_, dtype=np.float32).reshape(256, 4)
    ORB_PATTERN = np.vstack([ORB_PATTERN[:, :2], ORB_PATTERN[:, 2:]])  # Shape: (512, 2)

    def __init__(self, nfeatures=1000, scale_factor=1.2, nlevels=8, ini_th_FAST=20, min_th_FAST=7):
        self.nfeatures = nfeatures
        self.scale_factor = scale_factor
        self.nlevels = nlevels
        self.ini_th_FAST = ini_th_FAST
        self.min_th_FAST = min_th_FAST

        # Precompute scale factors and sigma
        self.mvScaleFactor = np.array([scale_factor**i for i in range(nlevels)], dtype=np.float32)
        self.mvLevelSigma2 = self.mvScaleFactor**2
        self.mvInvScaleFactor = 1.0 / self.mvScaleFactor
        self.mvInvLevelSigma2 = 1.0 / self.mvLevelSigma2
        self.mvImagePyramid = [None] * nlevels

        # Distribute features across levels
        factor = 1.0 / scale_factor
        nDesiredFeaturesPerScale = nfeatures * (1 - factor) / (1 - factor**nlevels)
        self.mnFeaturesPerLevel = np.round(nDesiredFeaturesPerScale * factor**np.arange(nlevels)).astype(int)
        self.mnFeaturesPerLevel[-1] = max(nfeatures - np.sum(self.mnFeaturesPerLevel[:-1]), 0)

        self.umax = self._initialize_umax()

    def _initialize_umax(self):
        # Precompute circle boundary for orientation
        umax = np.zeros(self.HALF_PATCH_SIZE + 1, dtype=np.int32)
        vmax = ceil(self.HALF_PATCH_SIZE * sqrt(2) / 2)
        hp2 = self.HALF_PATCH_SIZE**2
        v_range = np.arange(vmax + 1)
        umax[:vmax + 1] = np.round(np.sqrt(hp2 - v_range**2)).astype(int)
        # Mirror values for v > vmax
        for v in range(vmax, self.HALF_PATCH_SIZE + 1):
            umax[v] = umax[self.HALF_PATCH_SIZE - v]
        return umax

    def compute_pyramid(self, image):
        assert image.ndim == 2, "Image must be grayscale"
        h, w = image.shape
        self.mvImagePyramid[0] = image

        for level in range(1, self.nlevels):
            scale = self.mvInvScaleFactor[level]
            sz = (int(round(w * scale)), int(round(h * scale)))
            # Resize directly to target size, avoiding temporary buffer
            self.mvImagePyramid[level] = cv2.resize(
                image, sz, interpolation=cv2.INTER_LINEAR
            )

    def IC_Angle(self, image, pt):
        x, y = int(pt[0]), int(pt[1])
        hps = self.HALF_PATCH_SIZE
        patch = image[y - hps:y + hps + 1, x - hps:x + hps + 1]
        if patch.shape != (self.PATCH_SIZE, self.PATCH_SIZE):
            return 0.0

        m_01, m_10 = 0.0, 0.0
        center = hps

        # Compute moments using vectorized operations
        for v in range(1, hps + 1):
            d = self.umax[v]
            u_range = np.arange(-d, d + 1)
            val_plus = patch[center + v, center + u_range].astype(np.float32)
            val_minus = patch[center - v, center + u_range].astype(np.float32)
            m_01 += v * np.sum(val_plus - val_minus)
            m_10 += np.sum(u_range * (val_plus + val_minus))

        return atan2(m_01, m_10) * 180.0 / pi

    def compute_orientation(self, image, keypoints):
        angles = np.array([self.IC_Angle(image, kp.pt) for kp in keypoints], dtype=np.float32)
        for kp, angle in zip(keypoints, angles):
            kp.angle = angle

    def compute_orb_descriptor(self, keypoints, img):
        if not keypoints:
            return [], np.zeros((0, 32), dtype=np.uint8)

        h, w = img.shape
        descriptors = np.zeros((len(keypoints), 32), dtype=np.uint8)
        valid_kps = []
        valid_idx = []

        # Filter valid keypoints
        for idx, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if (self.HALF_PATCH_SIZE <= x < w - self.HALF_PATCH_SIZE and
                self.HALF_PATCH_SIZE <= y < h - self.HALF_PATCH_SIZE):
                valid_kps.append(kp)
                valid_idx.append(idx)

        if not valid_kps:
            return [], descriptors

        # Precompute rotations
        angles = np.radians([kp.angle for kp in valid_kps])
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        cx = np.array([kp.pt[0] for kp in valid_kps])
        cy = np.array([kp.pt[1] for kp in valid_kps])

        for i in range(32):
            for j in range(8):
                idx = i * 16 + 2 * j
                p1, p2 = self.ORB_PATTERN[idx], self.ORB_PATTERN[idx + 1]

                # Rotate points for all keypoints
                x1 = p1[0] * cos_a - p1[1] * sin_a + cx
                y1 = p1[0] * sin_a + p1[1] * cos_a + cy
                x2 = p2[0] * cos_a - p2[1] * sin_a + cx
                y2 = p2[0] * sin_a + p2[1] * cos_a + cy

                # Round to integers
                x1, y1 = np.round(x1).astype(int), np.round(y1).astype(int)
                x2, y2 = np.round(x2).astype(int), np.round(y2).astype(int)

                # Check bounds and compare intensities
                valid = (x1 >= 0) & (x1 < w) & (y1 >= 0) & (y1 < h) & \
                        (x2 >= 0) & (x2 < w) & (y2 >= 0) & (y2 < h)
                intensities1 = np.zeros_like(x1)
                intensities2 = np.zeros_like(x2)
                intensities1[valid] = img[y1[valid], x1[valid]]
                intensities2[valid] = img[y2[valid], x2[valid]]
                bits = (intensities1 < intensities2).astype(np.uint8) << j
                descriptors[valid_idx, i] |= bits

        return valid_kps, descriptors[np.array(valid_idx)]

    def __call__(self, image):
        self.compute_pyramid(image)
        keypoints_all = []
        descriptors_all = []

        # Initialize FAST detectors
        fast = cv2.FastFeatureDetector_create(threshold=self.ini_th_FAST, nonmaxSuppression=True)
        fast_low = cv2.FastFeatureDetector_create(threshold=self.min_th_FAST, nonmaxSuppression=True)

        for level in range(self.nlevels):
            img = self.mvImagePyramid[level]
            kps = fast.detect(img, None)
            
            # Fallback to lower threshold if needed
            if len(kps) < self.mnFeaturesPerLevel[level]:
                kps = fast_low.detect(img, None)
                kps = kps[:self.mnFeaturesPerLevel[level]]  # Limit keypoints

            # Adjust keypoint coordinates
            for kp in kps:
                kp.octave = level
                kp.pt = (kp.pt[0] * self.mvScaleFactor[level], 
                        kp.pt[1] * self.mvScaleFactor[level])

            self.compute_orientation(img, kps)
            kps, desc = self.compute_orb_descriptor(kps, img)
            
            keypoints_all.extend(kps)
            if desc.size > 0:
                descriptors_all.append(desc)

        descriptors_all = np.vstack(descriptors_all) if descriptors_all else np.zeros((0, 32), dtype=np.uint8)
        return keypoints_all, descriptors_all