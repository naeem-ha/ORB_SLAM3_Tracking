import cv2
import numpy as np
# import matplotlib.pyplot as plt
from src.ORBMatcher3D import ORBMatcher3D
# from src.ORB_Extractor import ORBExtractor
# # Load the images
from python_orb_slam3 import ORBExtractor as OGORBExtractor

source = cv2.imread("/home/hamza-naeem/Documents/ORB_SLAM3_Tracking/1403636579763555584.png", cv2.IMREAD_GRAYSCALE)
target = cv2.imread("/home/hamza-naeem/Documents/ORB_SLAM3_Tracking/1403636579813555456.png", cv2.IMREAD_GRAYSCALE)
import time

K = np.array([
    [458.654, 0, 367.215],
    [0, 457.296, 248.375],
    [0, 0, 1]
])
FAKE_DEPTH = 1.0

orb_slam3 = OGORBExtractor()

start = time.time()
kp1_s3, des1_s3 = orb_slam3.detectAndCompute(source)
kp2_s3, des2_s3 = orb_slam3.detectAndCompute(target)

print("Time for 2 extractions" + str (time.time() - start))
# --- Backproject keypoints from image 1 into 3D space (fake Z=1) ---
map_points = []
K_inv = np.linalg.inv(K)
for i, kp in enumerate(kp1_s3):
    if des1_s3 is None or i >= len(des1_s3):
        continue
    x, y = kp.pt
    p_img = np.array([x * FAKE_DEPTH, y * FAKE_DEPTH, FAKE_DEPTH])
    X = K_inv @ p_img
    mp = {
        'pos': X,
        'desc': des1_s3[i],
        'normal': np.array([0, 0, 1]),  # assume frontal normals
        'angle': kp.angle
    }
    map_points.append(mp)

# --- Camera pose: identity (matching between close images) ---
Rcw = np.eye(3)
tcw = np.zeros((3, 1))

# --- Initialize matcher ---
matcher = ORBMatcher3D(
    camera_matrix=K,
    scale_factors=[1.2 ** i for i in range(8)],
    check_orientation=True
)

# --- Perform matching ---
matches = matcher.match(
    map_points=map_points,
    keypoints=kp2_s3,
    descriptors=des2_s3,
    Rcw=Rcw,
    tcw=tcw
)

# --- Draw matches ---
matches = sorted(matches, key=lambda m: m.distance)
matched_img = cv2.drawMatches(source, kp1_s3, target, kp2_s3, matches[:50], None, flags=2)

# --- Save & Show ---
cv2.imwrite("OG_ORB-SLAM3_Feature_Matches.jpg", matched_img)






# bf_s3 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches_s3 = bf_s3.match(des1_s3, des2_s3)
# matches_s3 = sorted(matches_s3, key=lambda x: x.distance)

# matched_img_s3 = cv2.drawMatches(source, kp1_s3, target, kp2_s3, matches_s3[:50], None, flags=2)

# cv2.imwrite('ORB-SLAM3 Style Matcher.jpg', matched_img_s3)

# K = np.array([
#     [458.654, 0, 367.215],
#     [0, 457.296, 248.375],
#     [0, 0, 1]
# ])


# map_points = []
# for i, kp in enumerate(kp1_s3):
#     if des1_s3 is None or i >= len(des1_s3):
#         continue
#     x, y = kp.pt
#     z = 1.0  # fake depth assumption
#     # Backproject to 3D using K
#     X = np.linalg.inv(K) @ np.array([x * z, y * z, z])
#     mp = {
#         'pos': X,
#         'desc': des1_s3[i],
#         'normal': np.array([0, 0, 1]),  # assume facing forward
#         'angle': kp.angle
#     }
#     map_points.append(mp)

# frame_kps = kp2_s3
# frame_descs = des2_s3
# Rcw = np.eye(3)
# tcw = np.zeros((3, 1))


# matcher = ORBMatcher3D(
#     camera_matrix=K,
#     scale_factors=[1.0 * (1.2 ** i) for i in range(8)],
#     check_orientation=True
# )

# matches = matcher.match(
#     map_points=map_points,            # list of {pos, desc, normal, angle}
#     keypoints=frame_kps,              # list of cv2.KeyPoint
#     descriptors=frame_descs,          # np.ndarray Nx32
#     Rcw=Rcw,                          # 3x3 np.ndarray
#     tcw=tcw                           # 3x1 np.ndarray
# )


# # matches_s3_custom = orbslam3_matcher(des1_s3, des2_s3, kp1_s3, kp2_s3, ratio_thresh=0.75, orientation_check=True)
# matched_img = cv2.drawMatches(source, kp1_s3, target, frame_kps, matches[:50], None)
# cv2.imwrite('ORB-SLAM3 Style Matcher.jpg', matched_img)


# #plt.tight_layout()
# #plt.show()











# # # --- Method 1: OpenCV ORB ---
# # orb_opencv = orb_opencv = cv2.ORB_create(
# #     nfeatures=1000,
# #     scaleFactor=1.2,
# #     nlevels=8,
# #     edgeThreshold=31,
# #     firstLevel=0,
# #     WTA_K=2,
# #     scoreType=cv2.ORB_HARRIS_SCORE,
# #     patchSize=31,
# #     fastThreshold=20
# # )
# # kp1_cv, des1_cv = orb_opencv.detectAndCompute(source, None)
# # kp2_cv, des2_cv = orb_opencv.detectAndCompute(target, None)

# # bf_cv = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# # matches_cv = bf_cv.match(des1_cv, des2_cv)
# # matches_cv = sorted(matches_cv, key=lambda x: x.distance)

# # matched_img_cv = cv2.drawMatches(source, kp1_cv, target, kp2_cv, matches_cv[:50], None, flags=2)

# # # --- Method 2: ORB-SLAM3 ORBExtractor ---
# # orb_slam3 = ORBExtractor()
# # kp1_s3, des1_s3 = orb_slam3.detectAndCompute(source)
# # kp2_s3, des2_s3 = orb_slam3.detectAndCompute(target)

# # # Check if keypoints are converted properly
# # if isinstance(kp1_s3[0], tuple):  # If ORBExtractor returns (x, y), convert to cv2.KeyPoint
# #     kp1_s3 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=20) for pt in kp1_s3]
# #     kp2_s3 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=20) for pt in kp2_s3]

# # bf_s3 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# # matches_s3 = bf_s3.match(des1_s3, des2_s3)
# # matches_s3 = sorted(matches_s3, key=lambda x: x.distance)

# # matched_img_s3 = cv2.drawMatches(source, kp1_s3, target, kp2_s3, matches_s3[:50], None, flags=2)



# # cv2.imwrite('OpenCV ORB Matching.jpg', matched_img_cv)


# # cv2.imwrite('ORB-SLAM3 ORBExtractor Matching.jpg', matched_img_s3)



import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.ORB_Extractor import ORBExtractor
from src.ORBMatcher3D import ORBMatcher3D

# --- Configuration ---
K = np.array([
    [458.654, 0, 367.215],
    [0, 457.296, 248.375],
    [0, 0, 1]
])
FAKE_DEPTH = 1.0

# --- Load images ---
img1 = cv2.imread("/home/hamza-naeem/Documents/ORB_SLAM3_Tracking/1403636579763555584.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("/home/hamza-naeem/Documents/ORB_SLAM3_Tracking/1403636579813555456.png", cv2.IMREAD_GRAYSCALE)

# --- Initialize extractor ---
orb = ORBExtractor(nfeatures=1000, scale_factor=1.2, nlevels=8, ini_th_fast=20, min_th_fast=7)

# --- Extract features ---
start = time.time()
kp1, desc1, _ = orb(img1)
kp2, desc2, _ = orb(img2)
print("Time for 2 extractions" + str (time.time() - start))
# --- Backproject keypoints from image 1 into 3D space (fake Z=1) ---
map_points = []
K_inv = np.linalg.inv(K)
for i, kp in enumerate(kp1):
    if desc1 is None or i >= len(desc1):
        continue
    x, y = kp.pt
    p_img = np.array([x * FAKE_DEPTH, y * FAKE_DEPTH, FAKE_DEPTH])
    X = K_inv @ p_img
    mp = {
        'pos': X,
        'desc': desc1[i],
        'normal': np.array([0, 0, 1]),  # assume frontal normals
        'angle': kp.angle
    }
    map_points.append(mp)

# --- Camera pose: identity (matching between close images) ---
Rcw = np.eye(3)
tcw = np.zeros((3, 1))

# --- Initialize matcher ---
matcher = ORBMatcher3D(
    camera_matrix=K,
    scale_factors=[1.2 ** i for i in range(8)],
    check_orientation=True
)

# --- Perform matching ---
matches = matcher.match(
    map_points=map_points,
    keypoints=kp2,
    descriptors=desc2,
    Rcw=Rcw,
    tcw=tcw
)

# --- Draw matches ---
matches = sorted(matches, key=lambda m: m.distance)
matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

# --- Save & Show ---
cv2.imwrite("ORB-SLAM3_Feature_Matches.jpg", matched_img)
# plt.figure(figsize=(12, 6))
# plt.imshow(matched_img[..., ::-1])
# plt.title("Top 50 Matches (ORB-SLAM3 Style)")
# plt.axis("off")
# plt.tight_layout()
# plt.show()
