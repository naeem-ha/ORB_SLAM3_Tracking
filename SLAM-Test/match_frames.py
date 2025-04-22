from ORBMatcher3D import ORBMatcher3D  # Assuming it's defined in a file named ORBMatcher3D.py
import numpy as np
import cv2
np.set_printoptions(suppress=True)

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform

def extractRt(F):
  W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
  U,d,Vt = np.linalg.svd(F)
  #assert np.linalg.det(U) > 0
  if np.linalg.det(Vt) < 0:
    Vt *= -1.0
  R = np.dot(np.dot(U, W), Vt)
  if np.sum(R.diagonal()) < 0:
    R = np.dot(np.dot(U, W.T), Vt)
  t = U[:, 2]
  ret = np.eye(4)
  ret[:3, :3] = R
  ret[:3, 3] = t
  #print(d)
  return ret

def generate_match(f1, f2):
    # Initialize the matcher with example params (you can adjust)
    matcher = ORBMatcher3D(
        camera_matrix=np.eye(3),
        scale_factors=[1.0 * (1.2 ** i) for i in range(8)],
        check_orientation=True
    )

    # Use identity rotation and zero translation for relative matching
    Rcw = np.eye(3)
    tcw = np.zeros((3, 1))

    # Recreate 3D map points from f1 keypoints using fake depth = 1.0
    map_points = []
    for i, (x, y) in enumerate(f1.key_pts):
        z = 1.0
        X = np.array([x * z, y * z, z])
        mp = {
            'pos': X,
            'desc': f1.descriptors[i],
            'normal': np.array([0, 0, 1]),
            'angle': 0.0  # You can replace with real angle if available
        }
        map_points.append(mp)

    # Convert f2.key_pts (Nx2) to cv2.KeyPoint list
    keypoints_f2 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=20) for pt in f2.key_pts]

    # Run custom matcher
    matches = matcher.match(
        map_points=map_points,
        keypoints=keypoints_f2,
        descriptors=f2.descriptors,
        Rcw=Rcw,
        tcw=tcw
    )

    # Lowe-style validation using geometry and match uniqueness
    ret = []
    x1, x2 = [], []

    for m in matches:
        idx1 = m.queryIdx
        idx2 = m.trainIdx
        pts1 = f1.key_pts[idx1]
        pts2 = f2.key_pts[idx2]

        if np.linalg.norm((pts1 - pts2)) < 0.1 * np.linalg.norm([f1.w, f1.h]) and m.distance < 32:
            if idx1 not in x1 and idx2 not in x2:
                x1.append(idx1)
                x2.append(idx2)
                ret.append((pts1, pts2))

    assert len(set(x1)) == len(x1)
    assert len(set(x2)) == len(x2)
    assert len(ret) >= 8, "Not enough inliers after matching"

    ret = np.array(ret)
    x1 = np.array(x1)
    x2 = np.array(x2)

    # RANSAC to get F and extract Rt
    model, f_pts = ransac((ret[:, 0], ret[:, 1]),
                          FundamentalMatrixTransform,
                          min_samples=8,
                          residual_threshold=0.001,
                          max_trials=100)
    print("Matches: %d -> %d -> %d -> %d" % (
        len(f1.descriptors), len(matches), len(f_pts), sum(f_pts)
    ))

    Rt = extractRt(model.params)
    return x1[f_pts], x2[f_pts], Rt
