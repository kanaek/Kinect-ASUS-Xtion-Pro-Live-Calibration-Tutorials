#!/usr/bin/env python


import rospy
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import yaml
import sys
from matplotlib import pyplot as plt

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

stream = open("/home/chentao/kinect_calibration/rgb_1504270110.yaml", "r")
doc = yaml.load(stream)
rgb_mtx = np.array(doc['camera_matrix']['data']).reshape(3,3)
rgb_dist = np.array(doc['distortion_coefficients']['data'])

stream = open("/home/chentao/kinect_calibration/depth_1504270110.yaml", "r")
doc = yaml.load(stream)
depth_mtx = np.array(doc['camera_matrix']['data']).reshape(3,3)
depth_dist = np.array(doc['distortion_coefficients']['data'])

width = 0.0348
delta_x = 0    #0.0824
delta_y = 0    #0.064
x_num = 8
y_num = 6
objpoints = np.zeros((x_num * y_num, 3), np.float32)

axis = np.float32([[delta_x,delta_y,0],[delta_x + width,delta_y,0], [delta_x,delta_y + width,0], [delta_x,delta_y,delta_y + width]]).reshape(-1,3)
axis[:,0] += width * 5
axis[:,1] += width * 2

class image_converter:

    def __init__(self):

        self.br = CvBridge()
        # self.image_sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.rgb_callback)
        self.image_sub = rospy.Subscriber("/camera/ir/image_raw",Image,self.ir_callback)

    def rgb_callback(self,data):
        try:
            img = self.br.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # ret, corners = cv2.findChessboardCorners(gray, (x_num,y_num),None)
        ret, corners = cv2.findChessboardCorners(img, (x_num,y_num))
        cv2.imshow('img',img)
        cv2.waitKey(5)
        if ret == True:
            cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
            tempimg = img.copy()
            cv2.drawChessboardCorners(tempimg, (x_num,y_num), corners,ret)
            # ret, rvec, tvec = cv2.solvePnP(objpoints, corners, mtx, dist, flags = cv2.CV_EPNP)
            rvec, tvec, inliers = cv2.solvePnPRansac(objpoints, corners, rgb_mtx, rgb_dist)
            print("rvecs:")
            print(rvec)
            print("tvecs:")
            print(tvec)
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvec, tvec, rgb_mtx, rgb_dist)

            imgpts = np.int32(imgpts).reshape(-1,2)

            cv2.line(tempimg, tuple(imgpts[0]), tuple(imgpts[1]),[255,0,0],4)  #BGR
            cv2.line(tempimg, tuple(imgpts[0]), tuple(imgpts[2]),[0,255,0],4)
            cv2.line(tempimg, tuple(imgpts[0]), tuple(imgpts[3]),[0,0,255],4)

            cv2.imshow('img',tempimg)
            cv2.waitKey(5)

    def ir_callback(self,data):
        try:
          gray = self.mkgray(data)
        except CvBridgeError as e:
          print(e)

        # ret, corners = cv2.findChessboardCorners(gray, (x_num,y_num),None)
        ret, corners = cv2.findChessboardCorners(gray, (x_num,y_num))
        cv2.imshow('img',gray)
        cv2.waitKey(5)
        if ret == True:
            cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
            tempimg = gray.copy()
            cv2.drawChessboardCorners(tempimg, (x_num,y_num), corners,ret)
            # ret, rvec, tvec = cv2.solvePnP(objpoints, corners, mtx, dist, flags = cv2.CV_EPNP)
            rvec, tvec, inliers = cv2.solvePnPRansac(objpoints, corners, depth_mtx, depth_dist)
            print("rvecs:")
            print(rvec)
            print("tvecs:")
            print(tvec)
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvec, tvec, depth_mtx, depth_dist)

            imgpts = np.int32(imgpts).reshape(-1,2)

            cv2.line(tempimg, tuple(imgpts[0]), tuple(imgpts[1]),[255,0,0],4)  #BGR
            cv2.line(tempimg, tuple(imgpts[0]), tuple(imgpts[2]),[0,255,0],4)
            cv2.line(tempimg, tuple(imgpts[0]), tuple(imgpts[3]),[0,0,255],4)

            cv2.imshow('img',tempimg)
            cv2.waitKey(5)


    def mkgray(self, msg):
        """
        Convert a message into a 8-bit 1 channel monochrome OpenCV image
        """
        # as cv_bridge automatically scales, we need to remove that behavior
        # TODO: get a Python API in cv_bridge to check for the image depth.
        if self.br.encoding_to_dtype_with_channels(msg.encoding)[0] in ['uint16', 'int16']:

            mono16 = self.br.imgmsg_to_cv2(msg, '16UC1')
            mono8 = np.array(np.clip(mono16, 0, 255), dtype=np.uint8)
            return mono8
        elif 'FC1' in msg.encoding:
            # floating point image handling
            img = self.br.imgmsg_to_cv2(msg, "passthrough")
            _, max_val, _, _ = cv2.minMaxLoc(img)
            if max_val > 0:
                scale = 255.0 / max_val
                mono_img = (img * scale).astype(np.uint8)
            else:
                mono_img = img.astype(np.uint8)
            return mono_img
        else:
            return self.br.imgmsg_to_cv2(msg, "mono8")

if __name__ == "__main__":
    ic = image_converter()
    for i in range(y_num):
        for j in range(x_num):
            index = i * x_num + j
            objpoints[index,0] = delta_x + j * width
            objpoints[index,1] = delta_y + (y_num - 1 - i) * width
            objpoints[index,2] = 0
    print objpoints
    rospy.init_node('pose_estimation')

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
