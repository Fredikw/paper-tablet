import cv2
import utils
import os

from os import listdir, getcwd
from argparse import ArgumentParser

# python .\pivproject2021.py 1 "/templates/template1_manyArucos.png" "output_folder" "/dataset_task1" 4


'''
Command line arguments

'''

parser = ArgumentParser()
parser.add_argument("task")
parser.add_argument("path_to_template")         # "templates/template1_manyArucos.png"
parser.add_argument("path_to_output_folder")
parser.add_argument("arg1")                     # "/dataset_task1"
parser.add_argument("arg2")

args = vars(parser.parse_args())

if args["task"] == "1":

    '''
    Read rgb images

    '''

    img1 = cv2.imread(getcwd() + args["path_to_template"])

    img_folder = args["arg1"]

    for image in listdir(getcwd() + img_folder):

        img2 = cv2.imread(getcwd() + img_folder + "/" + image)

        '''
        Find corresponding ArUco markers in images

        '''

        aruco_dict  = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_50)
        arucoParams = cv2.aruco.DetectorParameters_create()

        corners_img1, ids_img1, rejected_img1 = cv2.aruco.detectMarkers(img1, aruco_dict, parameters=arucoParams)
        corners_img2, ids_img2, rejected_img2 = cv2.aruco.detectMarkers(img2, aruco_dict, parameters=arucoParams)

        corners_img1 = utils.sort_markers(ids_img1, corners_img1)
        corners_img2 = utils.sort_markers(ids_img2, corners_img2)
        destpts      = utils.format_detectMarkers_corners(corners_img1)
        srcpts	     = utils.format_detectMarkers_corners(corners_img2)

        '''
        Find homography between images

        '''

        M, mask = cv2.findHomography(srcpts, destpts)

        tf_img = cv2.warpPerspective(img2, M, (2339, 1654))

        '''
        Displaying images

        TODO reshape image to fit on screen
        '''

        # cv2.imshow('frame', img2)
        # cv2.imshow('frame1', tf_img)
        # cv2.waitKey(0)

        '''
        Save image to output folder

        '''

        path = args["path_to_output_folder"]
        cv2.imwrite(os.path.join(path , "tf_" + image), tf_img)