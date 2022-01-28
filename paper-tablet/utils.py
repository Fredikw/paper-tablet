import cv2
import os

import numpy as np

# Sort corners by id
# id and courner correspond on index
def sort_markers(ids, corners):
	ids = ids.tolist()
	lst = []
	for count, _ in enumerate(ids):
		idx = ids.index([count])
		lst.append(corners[idx])
	return tuple(lst)


# Input array of arrays
# Returns single array
def format_detectMarkers_corners(corners):

    entries = 32
    dim     = 2

    arr = np.empty([entries, dim], dtype="float32")
    arr_idx = 0
    for mark in corners:
        for corner in mark[0]:
            corner = np.array([corner])
            arr[arr_idx] = corner
            arr_idx += 1
    return arr


def get_keypoints(img1, img2):
    
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    return kp1, des1, kp2, des2


def get_matching_pints(kp1, des1, kp2, des2):

    # Match the keypoints between images
    index_params  = dict(algorithm = 1, trees = 5)
    search_params = dict(checks = 50)
    flann         = cv2.FlannBasedMatcher(index_params, search_params)
    matches       = flann.knnMatch(des1,des2,k=2)

    # store all the good matches
    good = get_good_matches(matches)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    return src_pts, dst_pts


def task2(img_temp, img1):

    kp1, des1, kp2, des2 = get_keypoints(img_temp, img1)
    src_pts, dst_pts     = get_matching_pints(kp1, des1, kp2, des2)

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,8.0)

    tf_img = cv2.warpPerspective(img1, M, (img_temp.shape[1], img_temp.shape[0]))

    return tf_img


def get_good_matches(matches, threshold = 0.4, MIN_NUMBER_OF_CORR = 20):

    good_matches = [m for m,n in matches if (m.distance < threshold * n.distance)]

    if len(good_matches) < MIN_NUMBER_OF_CORR:
        
        good_matches = get_good_matches(matches, threshold + 0.1)

    return good_matches