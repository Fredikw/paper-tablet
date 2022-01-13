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
Find corresponding ArUco markers in images

'''

aruco_dict  = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_50)
arucoParams = cv2.aruco.DetectorParameters_create()

corners_img1, ids_img1, rejected_img1 = cv2.aruco.detectMarkers(img1, aruco_dict, parameters=arucoParams)
corners_img2, ids_img2, rejected_img2 = cv2.aruco.detectMarkers(img2, aruco_dict, parameters=arucoParams)

corners_img1 = utils.sort_markers(ids_img1, corners_img1)
corners_img2 = utils.sort_markers(ids_img2, corners_img2)

destpts = utils.format_detectMarkers_corners(corners_img1)
srcpts	= utils.format_detectMarkers_corners(corners_img2)

'''
Find homography between images

'''

M, mask = cv2.findHomography(srcpts, destpts)

tf_img = cv2.warpPerspective(img2, M, (2339, 1654))

'''
Displaying images

TODO reshape image to fit on screen
'''
cv2.imshow('frame', img2)
cv2.imshow('frame1', tf_img)
cv2.waitKey(0)