import cv2
import utils
import os
import numpy as np

from os import listdir, getcwd
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim


# # Run the command
# # $python .\pivproject2021.py 1 "/templates/template1_manyArucos.png" "output_folder" "/dataset_task1" 4
# # $python .\pivproject2021.py 2 "/templates/template2_fewArucos.png" "output_folder" "/dataset_task2" 4
# # $python .\pivproject2021.py 3 "/templates/template2_fewArucos.png" "output_folder" "/dataset_task3_1" "/dataset_task3_2"


'''
Command line arguments

'''
parser = ArgumentParser()

parser.add_argument("task")
parser.add_argument("path_to_template")
parser.add_argument("path_to_output_folder")
parser.add_argument("arg1")
parser.add_argument("arg2")

args = vars(parser.parse_args())

# # For testing
# args = {
#   "task": "",
#   "path_to_output_folder" : ""
# }
# args["task"] = "4"

if args["task"] == "1":

    '''
    Read rgb images

    '''
    img1 = cv2.imread(getcwd() + args["path_to_template"])

    img_folder = args["arg1"]

    for image_name in listdir(getcwd() + img_folder):

        img2 = cv2.imread(getcwd() + img_folder + "/" + image_name)

        '''
        Find corresponding ArUco markers in images

        '''
        # Defining ArUco dictionary and parameters used for detection
        aruco_dict  = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_50)
        arucoParams = cv2.aruco.DetectorParameters_create()

        corners_img1, ids_img1, rejected_img1 = cv2.aruco.detectMarkers(img1, aruco_dict, parameters=arucoParams)
        corners_img2, ids_img2, rejected_img2 = cv2.aruco.detectMarkers(img2, aruco_dict, parameters=arucoParams)

        # Formatting marker corner points
        corners_img1 = utils.sort_markers(ids_img1, corners_img1)
        corners_img2 = utils.sort_markers(ids_img2, corners_img2)
        destpts      = utils.format_detectMarkers_corners(corners_img1)
        srcpts	     = utils.format_detectMarkers_corners(corners_img2)

        '''
        Find homography between images

        '''
        M, mask = cv2.findHomography(srcpts, destpts)

        # Transforming input image
        tf_img = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))

        '''
        Displaying images

        '''
        # cv2.imshow('frame', img2)
        # cv2.imshow('frame1', tf_img)
        # cv2.waitKey(0)

        '''
        Save image to output folder

        '''
        path = args["path_to_output_folder"]        
        cv2.imwrite(os.path.join(path , "tf_" + image_name), tf_img)


if args["task"] == "2":

    '''
    Read rgb images

    '''
    img1 = cv2.imread(getcwd() + args["path_to_template"])

    img_folder = args["arg1"]

    for image_name in listdir(getcwd() + img_folder):

        img2 = cv2.imread(getcwd() + img_folder + "/" + image_name)

        '''
        Keypoint detection and matching

        '''
        sift = cv2.SIFT_create()

        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        # Match the keypoints between images
        index_params  = dict(algorithm = 1, trees = 5)
        search_params = dict(checks = 50)
        flann         = cv2.FlannBasedMatcher(index_params, search_params)
        matches       = flann.knnMatch(des1,des2,k=2)

        # Get best matches
        good = utils.get_good_matches(matches)

        # Extract coordinates of matches found
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        '''
        Find homography between images

        '''
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)

        tf_img = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))

        '''
        Save image to output folder

        '''
        path = args["path_to_output_folder"]        
        cv2.imwrite(os.path.join(path , "tf_" + image_name), tf_img)

        # '''
        # Displaying inliners

        # '''

        # matchesMask = mask.ravel().tolist()
        # h,w, d = img1.shape
        # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # dst = cv2.perspectiveTransform(pts,M)
        
        # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        # draw_params = dict(matchColor = (0,255,0), singlePointColor = None, matchesMask = matchesMask, flags = 2)
        # img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        # plt.imshow(img3, 'gray'),plt.show()

'''
Approach based on ...

'''

if args["task"] == "3":

    img_temp         = cv2.imread(getcwd() + args["path_to_template"])
    img_temp_gray    = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)

    img1_folder = args["arg1"] #= "/FewArucos-Viewpoint1_images"
    img2_folder = args["arg2"] #= "/FewArucos-Viewpoint2_images"

    for image1_name, image2_name in zip(listdir(getcwd() + img1_folder), listdir(getcwd() + img2_folder)):

        img1 = cv2.imread(getcwd() + img1_folder + "/" + image1_name)
        img2 = cv2.imread(getcwd() + img2_folder + "/" + image2_name)

        tf_img1 = utils.task2(img_temp, img1)
        tf_img2 = utils.task2(img_temp, img2)

        '''
        Converting to gray scale image

        '''
        gray_img_1  = cv2.cvtColor(tf_img1, cv2.COLOR_BGR2GRAY)
        gray_img_2  = cv2.cvtColor(tf_img2, cv2.COLOR_BGR2GRAY)

        '''
        Normalize image

        '''
        gray_img_1_norm = np.zeros((img_temp.shape[0],img_temp.shape[1]))
        gray_img_2_norm = np.zeros((img_temp.shape[0],img_temp.shape[1]))

        gray_img_1_norm = cv2.normalize(gray_img_1, gray_img_1_norm, 0, 255, cv2.NORM_MINMAX)
        gray_img_2_norm = cv2.normalize(gray_img_2, gray_img_2_norm, 0, 255, cv2.NORM_MINMAX)


        '''
        Comparing images
        
        '''
        # # Comparing images using Structural Similarity Index (SSIM)
        # score_img1, diff_img1 = compare_ssim(img_temp, gray_img_1, full=True)
        # score_img2, diff_img2 = compare_ssim(img_temp, gray_img_2, full=True)

        # print("SSIM: {}".format(score_img1))
        # print("SSIM: {}".format(score_img2))

        # # Comparing images using Mean Squared Error
        err1 = np.sum((gray_img_1_norm.astype("float") - img_temp_gray.astype("float")) ** 2)
        err1 /= float(gray_img_1_norm.shape[0] * gray_img_1_norm.shape[1])

        err2 = np.sum((gray_img_2_norm.astype("float") - img_temp_gray.astype("float")) ** 2)
        err2 /= float(gray_img_2_norm.shape[0] * gray_img_2_norm.shape[1])

        '''
        Save images

        '''
        if err1 < err2:
            to_be_saved      = tf_img1
            to_be_saved_name = image1_name
        else:
            to_be_saved      = tf_img2
            to_be_saved_name = image2_name
        
        # Define image name
        path = args["path_to_output_folder"] #= "output_folder3" 
        cv2.imwrite(os.path.join(path , to_be_saved_name), to_be_saved)

# # for testing
if args["task"] == "4":
    pass