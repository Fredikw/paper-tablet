import numpy as np
import cv2
import utils

from matplotlib import pyplot as plt


'''
Read rgb images
TODO image path should be read as command line arguments

'''

img1 = cv2.imread('templates/template1_manyArucos.png')
img2 = cv2.imread('dataset/rgb_0720.jpg')


'''
Find ArUco markers in images

'''

aruco_dict  = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_50)
arucoParams = cv2.aruco.DetectorParameters_create()

corners_img1, ids_img1, rejected_img1 = cv2.aruco.detectMarkers(img1, aruco_dict, parameters=arucoParams)
corners_img2, ids_img2, rejected_img2 = cv2.aruco.detectMarkers(img2, aruco_dict, parameters=arucoParams)

corners_img1 = utils.sort_markers(ids_img1, corners_img1)
corners_img2 = utils.sort_markers(ids_img2, corners_img2)

corners_img1 = utils.format_detectMarkers_corners(corners_img1)
corners_img2 = utils.format_detectMarkers_corners(corners_img2)


'''
Find homography between images

'''

M, mask = cv2.findHomography(corners_img1, corners_img2)
matchesMask = mask.ravel().tolist()

# Have no idea what I am doing
h, w, d = img1.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)

img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)


'''
Plot image

'''

cv2.imshow('image', img2) 
cv2.waitKey(0)