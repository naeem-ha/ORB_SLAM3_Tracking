import cv2
import numpy as np
from math import sqrt, ceil, floor
from typing import List, Tuple, Optional

class ORBExtractor:
    PATCH_SIZE = 31
    HALF_PATCH_SIZE = 15
    EDGE_THRESHOLD = 19

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

    PATTERN = np.array(BIT_PATTERN_31_, dtype=np.float32).reshape(-1, 4)

    def __init__(self, nfeatures: int = 1500, scale_factor: float = 1.2, nlevels: int = 8,
                 ini_th_fast: int = 10, min_th_fast: int = 3):
        self.nfeatures = nfeatures
        self.scale_factor = scale_factor
        self.nlevels = nlevels
        self.ini_th_fast = ini_th_fast
        self.min_th_fast = min_th_fast

        self.mv_scale_factor = np.ones(nlevels, dtype=np.float32)
        self.mv_level_sigma2 = np.ones(nlevels, dtype=np.float32)
        self.mv_scale_factor[0] = 1.0
        self.mv_level_sigma2[0] = 1.0
        for i in range(1, nlevels):
            self.mv_scale_factor[i] = self.mv_scale_factor[i-1] * scale_factor
            self.mv_level_sigma2[i] = self.mv_scale_factor[i] ** 2

        self.mv_inv_scale_factor = 1.0 / self.mv_scale_factor
        self.mv_inv_level_sigma2 = 1.0 / self.mv_level_sigma2
        self.mv_image_pyramid = [None] * nlevels

        factor = 1.0 / scale_factor
        n_desired_features_per_scale = nfeatures * (1 - factor) / (1 - (factor ** nlevels))
        self.mn_features_per_level = np.zeros(nlevels, dtype=int)
        sum_features = 0
        for level in range(nlevels-1):
            self.mn_features_per_level[level] = round(n_desired_features_per_scale)
            sum_features += self.mn_features_per_level[level]
            n_desired_features_per_scale *= factor
        self.mn_features_per_level[nlevels-1] = max(nfeatures - sum_features, 0)
        print(f"Features per level: {self.mn_features_per_level}")

        self.umax = np.zeros(self.HALF_PATCH_SIZE + 1, dtype=np.int32)
        vmax = floor(self.HALF_PATCH_SIZE * sqrt(2.0) / 2 + 1)
        vmin = ceil(self.HALF_PATCH_SIZE * sqrt(2.0) / 2)
        hp2 = self.HALF_PATCH_SIZE ** 2
        for v in range(vmax + 1):
            self.umax[v] = round(sqrt(hp2 - v * v))
        v0 = 0
        for v in range(self.HALF_PATCH_SIZE, vmin-1, -1):
            while self.umax[v0] == self.umax[v0 + 1]:
                v0 += 1
            self.umax[v] = v0
            v0 += 1

    def compute_pyramid(self, image: np.ndarray):
        """Compute the image pyramid with border padding."""
        assert image.ndim == 2 and image.dtype == np.uint8, "Image must be grayscale uint8"
        h, w = image.shape

        for level in range(self.nlevels):
            scale = self.mv_inv_scale_factor[level]
            sz = (max(1, round(w * scale)), max(1, round(h * scale)))
            temp = np.zeros((sz[1] + 2 * self.EDGE_THRESHOLD, sz[0] + 2 * self.EDGE_THRESHOLD), dtype=np.uint8)

            if level == 0:
                temp[self.EDGE_THRESHOLD:self.EDGE_THRESHOLD+h,
                     self.EDGE_THRESHOLD:self.EDGE_THRESHOLD+w] = image
            else:
                prev_img = self.mv_image_pyramid[level-1][
                    self.EDGE_THRESHOLD:-self.EDGE_THRESHOLD,
                    self.EDGE_THRESHOLD:-self.EDGE_THRESHOLD]
                resized = cv2.resize(prev_img, (sz[0], sz[1]), interpolation=cv2.INTER_LINEAR)
                temp[self.EDGE_THRESHOLD:self.EDGE_THRESHOLD+sz[1],
                     self.EDGE_THRESHOLD:self.EDGE_THRESHOLD+sz[0]] = resized

            self.mv_image_pyramid[level] = cv2.copyMakeBorder(
                temp, self.EDGE_THRESHOLD, self.EDGE_THRESHOLD,
                self.EDGE_THRESHOLD, self.EDGE_THRESHOLD,
                cv2.BORDER_REFLECT_101
            )

    def ic_angle(self, image: np.ndarray, pt_x: float, pt_y: float, umax: np.ndarray, half_patch_size: int) -> float:
        """Compute the orientation of a keypoint."""
        m_01, m_10 = np.int32(0), np.int32(0)
        y, x = int(round(pt_y)), int(round(pt_x))
        h, w = image.shape

        if not (half_patch_size <= x < w - half_patch_size and
                half_patch_size <= y < h - half_patch_size):
            return 0.0

        for u in range(-half_patch_size, half_patch_size + 1):
            m_10 += np.int32(u * image[y, x + u])

        for v in range(1, half_patch_size + 1):
            v_sum = np.int32(0)
            d = umax[v]
            for u in range(-int(d), int(d) + 1):
                val_plus = image[y + v, x + u]
                val_minus = image[y - v, x + u]
                v_sum += np.int32(val_plus - val_minus)
                m_10 += np.int32(u * (val_plus + val_minus))
            m_01 += np.int32(v * v_sum)

        return np.arctan2(m_01, m_10) * 180.0 / np.pi

    def compute_orb_descriptors_batch(self, img: np.ndarray, kpts_x: np.ndarray, kpts_y: np.ndarray,
                                     angles: np.ndarray, pattern: np.ndarray, half_patch_size: int) -> np.ndarray:
        """Compute ORB descriptors for multiple keypoints using vectorization."""
        n_kpts = len(kpts_x)
        desc = np.zeros((n_kpts, 32), dtype=np.uint8)
        h, w = img.shape

        cos_a = np.cos(angles * np.pi / 180.0)
        sin_a = np.sin(angles * np.pi / 180.0)
        x_int = np.round(kpts_x).astype(np.int32)
        y_int = np.round(kpts_y).astype(np.int32)

        valid = ((x_int - half_patch_size >= 0) & (x_int + half_patch_size < w) &
                 (y_int - half_patch_size >= 0) & (y_int + half_patch_size < h))

        for i in range(32):
            pattern_offset = i * 8
            for j in range(8):
                idx = pattern_offset + j
                x1, y1, x2, y2 = pattern[idx]

                u1 = np.round(x1 * cos_a - y1 * sin_a).astype(np.int32)
                v1 = np.round(x1 * sin_a + y1 * cos_a).astype(np.int32)
                u2 = np.round(x2 * cos_a - y2 * sin_a).astype(np.int32)
                v2 = np.round(x2 * sin_a + y2 * cos_a).astype(np.int32)

                px1 = x_int + u1
                py1 = y_int + v1
                px2 = x_int + u2
                py2 = y_int + v2

                mask = (valid & (px1 >= 0) & (px1 < w) & (py1 >= 0) & (py1 < h) &
                        (px2 >= 0) & (px2 < w) & (py2 >= 0) & (py2 < h))

                t0 = np.zeros(n_kpts, dtype=np.uint8)
                t1 = np.zeros(n_kpts, dtype=np.uint8)
                t0[mask] = img[py1[mask], px1[mask]]
                t1[mask] = img[py2[mask], px2[mask]]
                bit_value = ((t0 < t1) << j).astype(np.uint8)
                desc[:, i] |= bit_value

        return desc

    class ExtractorNode:
        def __init__(self):
            self.ul = (0, 0)
            self.ur = (0, 0)
            self.bl = (0, 0)
            self.br = (0, 0)
            self.v_keys = []
            self.b_no_more = False

        def divide_node(self, n1: 'ORBExtractor.ExtractorNode', n2: 'ORBExtractor.ExtractorNode',
                       n3: 'ORBExtractor.ExtractorNode', n4: 'ORBExtractor.ExtractorNode'):
            half_x = ceil((self.ur[0] - self.ul[0]) / 2)
            half_y = ceil((self.br[1] - self.ul[1]) / 2)

            n1.ul = self.ul
            n1.ur = (self.ul[0] + half_x, self.ul[1])
            n1.bl = (self.ul[0], self.ul[1] + half_y)
            n1.br = (self.ul[0] + half_x, self.ul[1] + half_y)

            n2.ul = n1.ur
            n2.ur = self.ur
            n2.bl = n1.br
            n2.br = (self.ur[0], self.ul[1] + half_y)

            n3.ul = n1.bl
            n3.ur = n1.br
            n3.bl = self.bl
            n3.br = (n1.br[0], self.bl[1])

            n4.ul = n3.ur
            n4.ur = n2.br
            n4.bl = n3.br
            n4.br = self.br

            for kp in self.v_keys:
                if kp.pt[0] < n1.ur[0]:
                    if kp.pt[1] < n1.br[1]:
                        n1.v_keys.append(kp)
                    else:
                        n3.v_keys.append(kp)
                elif kp.pt[1] < n1.br[1]:
                    n2.v_keys.append(kp)
                else:
                    n4.v_keys.append(kp)

            if len(n1.v_keys) == 1:
                n1.b_no_more = True
            if len(n2.v_keys) == 1:
                n2.b_no_more = True
            if len(n3.v_keys) == 1:
                n3.b_no_more = True
            if len(n4.v_keys) == 1:
                n4.b_no_more = True

    def distribute_oct_tree(self, v_to_distribute_keys: List[cv2.KeyPoint], min_x: int, max_x: int,
                           min_y: int, max_y: int, n: int, level: int) -> List[cv2.KeyPoint]:
        n_ini = max(1, round((max_x - min_x) / (max_y - min_y)))
        h_x = (max_x - min_x) / n_ini

        nodes = []
        for i in range(n_ini):
            node = self.ExtractorNode()
            node.ul = (round(h_x * i), 0)
            node.ur = (round(h_x * (i + 1)), 0)
            node.bl = (node.ul[0], max_y - min_y)
            node.br = (node.ur[0], max_y - min_y)
            nodes.append(node)

        for kp in v_to_distribute_keys:
            idx = int(kp.pt[0] / h_x)
            if 0 <= idx < n_ini:
                nodes[idx].v_keys.append(kp)

        nodes = [node for node in nodes if node.v_keys]
        for node in nodes:
            if len(node.v_keys) == 1:
                node.b_no_more = True

        b_finish = False
        size_and_nodes = []

        while not b_finish:
            prev_size = len(nodes)
            size_and_nodes.clear()
            n_to_expand = 0

            new_nodes = []
            for node in nodes:
                if node.b_no_more:
                    new_nodes.append(node)
                    continue

                n1, n2, n3, n4 = self.ExtractorNode(), self.ExtractorNode(), self.ExtractorNode(), self.ExtractorNode()
                node.divide_node(n1, n2, n3, n4)

                if n1.v_keys:
                    new_nodes.append(n1)
                    if len(n1.v_keys) > 1:
                        n_to_expand += 1
                        size_and_nodes.append((len(n1.v_keys), n1))
                if n2.v_keys:
                    new_nodes.append(n2)
                    if len(n2.v_keys) > 1:
                        n_to_expand += 1
                        size_and_nodes.append((len(n2.v_keys), n2))
                if n3.v_keys:
                    new_nodes.append(n3)
                    if len(n3.v_keys) > 1:
                        n_to_expand += 1
                        size_and_nodes.append((len(n3.v_keys), n3))
                if n4.v_keys:
                    new_nodes.append(n4)
                    if len(n4.v_keys) > 1:
                        n_to_expand += 1
                        size_and_nodes.append((len(n4.v_keys), n4))

            nodes = new_nodes

            if len(nodes) >= n or len(nodes) == prev_size:
                b_finish = True
            elif len(nodes) + n_to_expand * 3 > n:
                while not b_finish:
                    prev_size = len(nodes)
                    prev_size_and_nodes = size_and_nodes.copy()
                    size_and_nodes.clear()

                    prev_size_and_nodes.sort(key=lambda x: (x[0], x[1].ul[0]), reverse=True)

                    new_nodes = [node for node in nodes]
                    for size, node in prev_size_and_nodes:
                        n1, n2, n3, n4 = self.ExtractorNode(), self.ExtractorNode(), self.ExtractorNode(), self.ExtractorNode()
                        node.divide_node(n1, n2, n3, n4)

                        new_nodes = [n for n in new_nodes if n is not node]
                        if n1.v_keys:
                            new_nodes.append(n1)
                            if len(n1.v_keys) > 1:
                                size_and_nodes.append((len(n1.v_keys), n1))
                        if n2.v_keys:
                            new_nodes.append(n2)
                            if len(n2.v_keys) > 1:
                                size_and_nodes.append((len(n2.v_keys), n2))
                        if n3.v_keys:
                            new_nodes.append(n3)
                            if len(n3.v_keys) > 1:
                                size_and_nodes.append((len(n3.v_keys), n3))
                        if n4.v_keys:
                            new_nodes.append(n4)
                            if len(n4.v_keys) > 1:
                                size_and_nodes.append((len(n4.v_keys), n4))

                        nodes = new_nodes
                        if len(nodes) >= n:
                            break

                    if len(nodes) >= n or len(nodes) == prev_size:
                        b_finish = True

        result_keys = []
        for node in nodes:
            if not node.v_keys:
                continue
            best_kp = max(node.v_keys, key=lambda kp: kp.response, default=node.v_keys[0])
            result_keys.append(best_kp)

        return result_keys

    def compute_keypoints_oct_tree(self) -> List[List[cv2.KeyPoint]]:
        all_keypoints = [[] for _ in range(self.nlevels)]
        W = 35.0

        for level in range(self.nlevels):
            min_border_x = self.EDGE_THRESHOLD - 3
            min_border_y = min_border_x
            max_border_x = self.mv_image_pyramid[level].shape[1] - self.EDGE_THRESHOLD + 3
            max_border_y = self.mv_image_pyramid[level].shape[0] - self.EDGE_THRESHOLD + 3

            v_to_distribute_keys = []
            width = max_border_x - min_border_x
            height = max_border_y - min_border_y
            n_cols = max(1, width / W)
            n_rows = max(1, height / W)
            w_cell = ceil(width / n_cols)
            h_cell = ceil(height / n_rows)

            fast = cv2.FastFeatureDetector_create(threshold=self.ini_th_fast, nonmaxSuppression=True)
            fast_low = cv2.FastFeatureDetector_create(threshold=self.min_th_fast, nonmaxSuppression=True)

            for i in range(int(n_rows)):
                ini_y = min_border_y + i * h_cell
                max_y = min(ini_y + h_cell + 6, max_border_y)
                if ini_y >= max_border_y - 3:
                    continue

                for j in range(int(n_cols)):
                    ini_x = min_border_x + j * w_cell
                    max_x = min(ini_x + w_cell + 6, max_border_x)
                    if ini_x >= max_border_x - 6:
                        continue

                    cell_img = self.mv_image_pyramid[level][int(ini_y):int(max_y), int(ini_x):int(max_x)]
                    kps = fast.detect(cell_img, None)

                    if not kps:
                        kps = fast_low.detect(cell_img, None)

                    if kps:
                        for kp in kps:
                            kp.pt = (kp.pt[0] + ini_x, kp.pt[1] + ini_y)
                            v_to_distribute_keys.append(kp)

            keypoints = self.distribute_oct_tree(
                v_to_distribute_keys, min_border_x, max_border_x,
                min_border_y, max_border_y, self.mn_features_per_level[level], level
            )

            scaled_patch_size = self.PATCH_SIZE * self.mv_scale_factor[level]
            for kp in keypoints:
                kp.pt = (kp.pt[0] + min_border_x, kp.pt[1] + min_border_y)
                kp.octave = level
                kp.size = scaled_patch_size

            all_keypoints[level] = keypoints

        for level in range(self.nlevels):
            img = self.mv_image_pyramid[level]
            img_blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
            for kp in all_keypoints[level]:
                kp.angle = self.ic_angle(
                    img_blurred, kp.pt[0], kp.pt[1], self.umax, self.HALF_PATCH_SIZE
                )

        return all_keypoints

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None,
                 v_lapping_area: Optional[List[int]] = None) -> Tuple[List[cv2.KeyPoint], np.ndarray, int]:
        if image.size == 0:
            return [], np.zeros((0, 32), dtype=np.uint8), -1
        print("Image is " + str(image.ndim) + str(image.dtype))
        assert image.ndim == 2 and image.dtype == np.uint8, "Image must be grayscale uint8"
        if mask is not None:
            assert mask.shape == image.shape and mask.dtype == np.uint8, "Invalid mask"

        self.compute_pyramid(image)
        all_keypoints = self.compute_keypoints_oct_tree()

        n_keypoints = sum(len(kps) for kps in all_keypoints)
        if n_keypoints == 0:
            return [], np.zeros((0, 32), dtype=np.uint8), 0

        keypoints = []
        descriptors = np.zeros((n_keypoints, 32), dtype=np.uint8)
        offset = 0
        mono_index = 0
        stereo_index = n_keypoints - 1

        for level in range(self.nlevels):
            kps = all_keypoints[level]
            if not kps:
                continue

            working_mat = self.mv_image_pyramid[level].copy()
            cv2.GaussianBlur(working_mat, (7, 7), 2, dst=working_mat, borderType=cv2.BORDER_REFLECT_101)

            kpts_x = np.array([kp.pt[0] for kp in kps], dtype=np.float32)
            kpts_y = np.array([kp.pt[1] for kp in kps], dtype=np.float32)
            angles = np.array([kp.angle for kp in kps], dtype=np.float32)
            desc = self.compute_orb_descriptors_batch(
                working_mat, kpts_x, kpts_y, angles, self.PATTERN, self.HALF_PATCH_SIZE
            )

            n_kps_level = len(kps)
            offset += n_kps_level

            if level != 0:
                scale = self.mv_scale_factor[level]
                for kp in kps:
                    kp.pt = (kp.pt[0] * scale, kp.pt[1] * scale)

            for i, kp in enumerate(kps):
                if v_lapping_area and v_lapping_area[0] <= kp.pt[0] <= v_lapping_area[1]:
                    keypoints.insert(stereo_index, kp)
                    descriptors[stereo_index] = desc[i]
                    stereo_index -= 1
                else:
                    keypoints.insert(mono_index, kp)
                    descriptors[mono_index] = desc[i]
                    mono_index += 1

        return keypoints, descriptors, mono_index