import cv2
import utils
import os
import numpy as np

from os import listdir, getcwd
from argparse import ArgumentParser
from matplotlib import pyplot as plt


# Run the command
# $python .\pivproject2021.py 1 "/templates/template1_manyArucos.png" "output_folder" "/dataset_task1" 4


# '''
# Command line arguments

# '''

# parser = ArgumentParser()
# parser.add_argument("task")
# parser.add_argument("path_to_template")         # "templates/template1_manyArucos.png"
# parser.add_argument("path_to_output_folder")
# parser.add_argument("arg1")                     # "/dataset_task1"
# parser.add_argument("arg2")

# args = vars(parser.parse_args())

args = {
  "task": "2"
}

args["task"] = "2"

# if args["task"] == "1":

#     '''
#     Read rgb images

#     '''

#     img1 = cv2.imread(getcwd() + args["path_to_template"])

#     img_folder = args["arg1"]

#     for image_name in listdir(getcwd() + img_folder):

#         img2 = cv2.imread(getcwd() + img_folder + "/" + image_name)

#         '''
#         Find corresponding ArUco markers in images

#         '''

#         # Defining ArUco dictionary and parameters used for detection
#         aruco_dict  = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_50)
#         arucoParams = cv2.aruco.DetectorParameters_create()

#         corners_img1, ids_img1, rejected_img1 = cv2.aruco.detectMarkers(img1, aruco_dict, parameters=arucoParams)
#         corners_img2, ids_img2, rejected_img2 = cv2.aruco.detectMarkers(img2, aruco_dict, parameters=arucoParams)

#         # Formatting marker corner points
#         corners_img1 = utils.sort_markers(ids_img1, corners_img1)
#         corners_img2 = utils.sort_markers(ids_img2, corners_img2)
#         destpts      = utils.format_detectMarkers_corners(corners_img1)
#         srcpts	     = utils.format_detectMarkers_corners(corners_img2)

#         '''
#         Find homography between images

#         '''

#         M, mask = cv2.findHomography(srcpts, destpts)

#         # Transforming input image
#         tf_img = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))

#         '''
#         Displaying images

#         '''

#         # cv2.imshow('frame', img2)
#         # cv2.imshow('frame1', tf_img)
#         # cv2.waitKey(0)

#         '''
#         Save image to output folder

#         '''

#         path = args["path_to_output_folder"]        
#         cv2.imwrite(os.path.join(path , "tf_" + image_name), tf_img)


if args["task"] == "2":

    img1 = cv2.imread("templates/template_glass.jpg")
    img2 = cv2.imread("dataset_task2/rgb4237.jpg")

    '''
    Find keypoints
    
    I have no idea what I am doing
    
    '''

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)


    # Match the keypoints between images
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:                    # if m.distance < 0.7*n.distance:
            good.append(m)                                  # Result in 19 matches

    # Extract coordinates of matches found
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)


    # M, mask = cv2.findHomography(src_pts, dst_pts)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)

    # tf_img = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))

    # path = getcwd() + "\output_folder"

    # cv2.imwrite(os.path.join(path , "tf_" + "rgb4237.jpg"), tf_img)

    matchesMask = mask.ravel().tolist()
    h,w, d = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    plt.imshow(img3, 'gray'),plt.show()


# if args["task"] == "3":
#     pass


# if args["task"] == "4":
#     pass