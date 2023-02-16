from lib.ORB_SLAM2 import orb_extractor_wrapper, test_size, orb_frame_matcher_wrapper
import numpy as np
import ctypes
import cv2

import sys,os
import time 

sys.path.append('/ORB_SLAM2')

img_paths = ['img.jpeg', 'img2.jpeg']
img_paths = ['/root/embedded_ads/embedded_ads_frames/1540019057_1010_00084380.png', '/root/embedded_ads/embedded_ads_frames/1540019057_1010_00084381.png']
# imgs = [cv2.imread(f)[:850, 250:] for f in imgs]  # remove logo and scoreboard regions 
default_size = (640, 480)
imgs = [] 
for f in img_paths:
    im = cv2.imread(f)[:850, 250:]
    im = cv2.resize(im, default_size, interpolation = cv2.INTER_AREA)
    imgs.append(im)
cv2.imwrite('/guanqing_ORB_SLAM2/dummy.png', im)

'''
orb extractor test 
tuned iniThFAST, minThFAST thresholds 
'''
test_extractor = 0
if test_extractor: 
    cols, rows = default_size
    ks = []
    ds = []
    n_kp = 1000
    iniThFAST = 20
    minThFAST = 10
    for img in imgs: 
        #print(test_size(img))

        k = np.zeros((n_kp, 2), dtype=np.float32)
        d = np.zeros((n_kp, 256), dtype=np.uint8)
        # orb_extractor_wrapper(np.ctypeslib.as_ctypes(img).data, rows, cols, k, d)
        # orb_extractor_wrapper(img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), rows, cols, k, d)
        r = orb_extractor_wrapper(np.uint8(img), rows, cols, n_kp, iniThFAST, minThFAST, k, d)

        ks.append(k)
        ds.append(d)

    k0 = [cv2.KeyPoint(x[0], x[1], 0) for x in ks[0]]
    k1 = [cv2.KeyPoint(x[0], x[1], 0) for x in ks[1]]
    im = cv2.drawKeypoints(imgs[0], k0, 0, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(f'/root/guanqing_ORB_SLAM2/keypoints0.png', im)
    im = cv2.drawKeypoints(imgs[1], k1, 0, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(f'/root/guanqing_ORB_SLAM2/keypoints1.png', im)

    ## keypoints for pyramid may contian level info 
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(ds[0], ds[1])
    MAX_DISTANCE = 730 # check mean and std of distance to determine 
    matches = [x for x in matches if x.distance <= MAX_DISTANCE]
    matches = sorted(matches, key=lambda x: x.distance)

    im = cv2.drawMatches(imgs[0], k0, imgs[1], k1, matches, 0, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('/root/guanqing_ORB_SLAM2/matches.png', im)

    # homography 
    src_pts = src_pts = np.float32([k0[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([k1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    im_src = cv2.warpPerspective(imgs[0], homography, (imgs[0].shape[1],imgs[0].shape[0]))
    overlay_input = (im_src * 0.5 + imgs[1] * 0.5).astype("uint8")
    cv2.imwrite('/root/guanqing_ORB_SLAM2/pybind11_bfmatch_warp_kp0.png', overlay_input)

n_kp = 1000 # 2000
iniThFAST = 20
minThFAST = 10
checkOri = False # always false, hardcoded in c++, 不需要check旋转不变性，因为相机不会旋转
window_size = 100 # 20
matcher_threshold = 50 # 150 # default 50 
cols, rows = default_size
kps = np.zeros((n_kp, 2, 2), dtype=np.float32)
matches = np.ones((n_kp, 1), dtype=np.intc) * -1

t1 = time.perf_counter()
n = 1000
for _ in range(n):
    nmatch = orb_frame_matcher_wrapper(imgs[0], imgs[1], rows, cols, n_kp, iniThFAST, minThFAST, window_size, matcher_threshold, kps, matches)
print(time.perf_counter() - t1, 'average call time ', (time.perf_counter() - t1) / n)
print('##### nmatches=', nmatch)

k0 = [cv2.KeyPoint(x[0], x[1], 0) for x in kps[:, 0, :]]
k1 = [cv2.KeyPoint(x[0], x[1], 0) for x in kps[:, 1, :]]
k0_matched = [cv2.KeyPoint(x[0], x[1], 0) for i, x in enumerate(kps[:, 0, :]) if matches[i]!=-1]
k1_matched = [cv2.KeyPoint(x[0], x[1], 0) for i, x in enumerate(kps[:, 1, :]) if matches[i]!=-1]

im = cv2.drawKeypoints(imgs[0], k0_matched, 0, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('/root/guanqing_ORB_SLAM2/pybind_slam_ORBmatch_kp0-matched.png', im)
im = cv2.drawKeypoints(imgs[1], k1_matched, 0, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('/root/guanqing_ORB_SLAM2/pybind_slam_ORBmatch_kp1-matched.png', im)
Dmatches = [cv2.DMatch(_imgIdx=0, _queryIdx=i, _trainIdx=int(matches[i]), _distance=0) for i in range(len(k0)) if matches[i]!=-1]
im = cv2.drawMatches(imgs[0], k0, imgs[1], k1, Dmatches, 0, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('/root/guanqing_ORB_SLAM2/pybind_slam_ORBmatch_matches.png', im)


# homography 
match_pts1 = np.float32([kps[:, 0, :][m.queryIdx] for m in Dmatches]).reshape(-1,1,2)
match_pts2 = np.float32([kps[:, 1, :][m.trainIdx] for m in Dmatches]).reshape(-1,1,2)
homography, _ = cv2.findHomography(match_pts1, match_pts2, cv2.RANSAC, 5.0)
im_src = cv2.warpPerspective(imgs[0], homography, (imgs[0].shape[1],imgs[0].shape[0]))
im_src = im_src*0.5 + imgs[1] * 0.5
cv2.imwrite('/root/guanqing_ORB_SLAM2/pybind_slam_ORBmatch_warp_overlay.png', im_src)

print('passed test')


# ## opencv baseline 
# orb = cv2.ORB_create(
#   nfeatures = 500,                    # The maximum number of features to retain.
#   scaleFactor = 1.2,                  # Pyramid decimation ratio, greater than 1
#   nlevels = 8,                        # The number of pyramid levels.
#   edgeThreshold = 7,                  # This is size of the border where the features are not detected. It should roughly match the patchSize parameter
#   firstLevel = 0,                     # It should be 0 in the current implementation.
#   WTA_K = 2,                          # The number of points that produce each element of the oriented BRIEF descriptor.
#   scoreType = cv2.ORB_HARRIS_SCORE,   # The default HARRIS_SCORE means that Harris algorithm is used to rank features (the score is written to KeyPoint::score and is 
#                                       # used to retain best nfeatures features); FAST_SCORE is alternative value of the parameter that produces slightly less stable 
#                                       # keypoints, but it is a little faster to compute.
#   #scoreType = cv2.ORB_FAST_SCORE,
#   patchSize = 7                       # size of the patch used by the oriented BRIEF descriptor. Of course, on smaller pyramid layers the perceived image area covered
#                                       # by a feature will be larger.
# )
# k0_cv, d0_cv = orb_cv.detectAndCompute(imgs[0], None)
# im = cv2.drawKeypoints(imgs[0], k0_cv, 0, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# cv2.imwrite('/root/guanqing_ORB_SLAM2/matcher_kp1_cv.png', im)