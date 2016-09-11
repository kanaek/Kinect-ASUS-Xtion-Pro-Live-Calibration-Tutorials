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

width = 0.0237
delta_x = 0.0426
delta_y = 0.0564
x_num = 6
y_num = 8
objpoints = np.zeros((x_num * y_num, 3), np.float32)

class registration:
    def __init__(self):

        self.br = CvBridge()
        #self.depth_image_sub = rospy.Subscriber("/camera/depth/image_raw",Image,self.depth_callback)

        # As you cannot get rgb and ir images simultaneously from kinect, you will need to get the coordinates 
        # of the corners separately on rgb image and ir image
        # To do this, uncomment the following two lines one at a time
        #self.rgb_image_sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.rgb_calib_callback)
        #self.ir_image_sub = rospy.Subscriber("/camera/ir/image_raw",Image,self.ir_calib_callback)
        self.ir_img = None
        self.rgb_img = None

        self.rgb_rmat = None
        self.rgb_tvec = None
        self.ir_rmat = None
        self.ir_tvec = None

        self.ir_to_rgb_rmat = None
        self.ir_to_rgb_tvec = None
        self.depth_image = None
        self.rgb_image = None
        self.rgb_corners = None
        self.ir_corners = None

        self.load_intrinsics()
        self.load_corners()
    
    def depth_callback(self,data):
    	try:
    		self.depth_image= self.br.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            print(e)


    def ir_calib_callback(self,data):
        try:
            self.ir_img = self.mkgray(data)
        except CvBridgeError as e:
            print(e)

        ir_ret, ir_corners = cv2.findChessboardCorners(self.ir_img, (y_num,x_num))
        cv2.imshow('ir_img',self.ir_img)
        cv2.waitKey(5)
        if ir_ret == True:
            ir_tempimg = self.ir_img.copy()
            cv2.cornerSubPix(ir_tempimg,ir_corners,(11,11),(-1,-1),criteria)            
            cv2.drawChessboardCorners(ir_tempimg, (y_num,x_num), ir_corners,ir_ret)
            # ret, rvec, tvec = cv2.solvePnP(objpoints, corners, mtx, dist, flags = cv2.CV_EPNP)

            depth_stream = open("/home/chentao/kinect_calibration/ir_camera_corners.yaml", "w")
            data = {'corners':ir_corners.tolist()}
            yaml.dump(data, depth_stream)

            cv2.imshow('ir_img',ir_tempimg)
            cv2.waitKey(5)

    def rgb_calib_callback(self,data):
        try:
            self.rgb_img = self.br.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        gray = cv2.cvtColor(self.rgb_img,cv2.COLOR_BGR2GRAY)
        rgb_ret, rgb_corners = cv2.findChessboardCorners(gray, (y_num,x_num),None)

        cv2.imshow('rgb_img',self.rgb_img)
        cv2.waitKey(5)
        if rgb_ret == True:
            rgb_tempimg = self.rgb_img.copy()
            cv2.cornerSubPix(gray,rgb_corners,(11,11),(-1,-1),criteria)            
            cv2.drawChessboardCorners(rgb_tempimg, (y_num,x_num), rgb_corners,rgb_ret)

            rgb_stream = open("/home/chentao/kinect_calibration/rgb_camera_corners.yaml", "w")
            data = {'corners':rgb_corners.tolist()}
            yaml.dump(data, rgb_stream)

            cv2.imshow('rgb_img',rgb_tempimg)
            cv2.waitKey(5)

    def load_intrinsics(self):
        ir_stream = open("/home/chentao/kinect_calibration/depth_1504270110.yaml", "r")
        ir_doc = yaml.load(ir_stream)
        self.ir_mtx = np.array(ir_doc['camera_matrix']['data']).reshape(3,3)
        self.ir_dist = np.array(ir_doc['distortion_coefficients']['data'])
        ir_stream.close()

        rgb_stream = open("/home/chentao/kinect_calibration/rgb_1504270110.yaml", "r")
        rgb_doc = yaml.load(rgb_stream)
        self.rgb_mtx = np.array(rgb_doc['camera_matrix']['data']).reshape(3,3)
        self.rgb_dist = np.array(rgb_doc['distortion_coefficients']['data'])
        rgb_stream.close()

    def load_corners(self):
        ir_stream = open("/home/chentao/kinect_calibration/ir_camera_corners.yaml", "r")
        ir_doc = yaml.load(ir_stream)
        self.ir_corners = np.array(ir_doc['corners']).reshape(-1,2).astype('float32')
        ir_stream.close()

        rgb_stream = open("/home/chentao/kinect_calibration/rgb_camera_corners.yaml", "r")
        rgb_doc = yaml.load(rgb_stream)
        self.rgb_corners = np.array(rgb_doc['corners']).reshape(-1,2)
        self.rgb_corners = self.rgb_corners.astype('float32')
        rgb_stream.close()


    def ir_to_rgb(self):
        if self.rgb_corners != None and self.ir_corners != None:

            _,_,_,_,_,R,T,E,F = cv2.stereoCalibrate([objpoints], [self.ir_corners], [self.rgb_corners],(480, 640), self.ir_mtx, self.ir_dist, self.rgb_mtx, self.rgb_dist, flags = cv2.cv.CV_CALIB_FIX_INTRINSIC)
            print "R:"
            print R
            print "T:"
            print T

    def register_depth_to_rgb(self):
    	if self.depth_image == None or self.rgb_image == None:
    		return

    	self.registered_depth = np.zeros(self.depth_image.shape)
    	self.registered_rgb = np.zeros(self.rgb_image.shape)
    	for row_i in range(self.depth_image.shape[0]):
    		for column_j in range(self.depth_image.shape[1]):
    			depth_pix_point = np.array([column_j, row_i, self.depth_image[row_i, column_j]])   #(x,y,z)
    			depth_coord_point = np.dot(np.linalg.inv(self.depth_mtx), depth_pix_point)
    			rgb_coord_point = np.dot(self.ir_to_rgb_rmat, depth_coord_point).reshape(3,1) + self.ir_to_rgb_tvec
    			rgb_pix_point = np.dot(self.rgb_mtx, rgb_coord_point)
    			rgb_x = int(rgb_pix_point[0] / float(rgb_pix_point[2]))
    			rgb_y = int(rgb_pix_point[1] / float(rgb_pix_point[2]))
    			rgb_x = np.clip(rgb_x, 0, self.rgb_image.shape[1] - 1)
                rgb_y = np.clip(rgb_y, 0, self.rgb_image.shape[0] - 1)
                self.registered_rgb[row_i, column_j, :] = self.rgb_image[rgb_y, rgb_x, :]
        cv2.imshow("Registered_rgb", self.registered_rgb)
        cv2.imshow("RGB", self.rgb_image)
        cv2.imshow("Depth",self.depth_image)
        cv2.waitKey(10)





    def get_ir_to_rgb_rmat(self):
        if self.ir_to_rgb_rmat != None:
            return self.ir_to_rgb_rmat


    def get_ir_to_rgb_tvec(self):
        if self.ir_to_rgb_tvec != None:
            return self.ir_to_rgb_tvec

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
    for i in range(x_num):
        for j in range(y_num):
            index = i * y_num + j
            objpoints[index,0] = delta_x + i * width
            objpoints[index,1] = delta_y + j * width
            objpoints[index,2] = 0
    rospy.init_node('pose_estimation')
    ic = registration()
    ic.ir_to_rgb()
    # rospy.spin()
    # try:
    #     ic.ir_to_rgb()
    #     print "=================================="
    #     print "rmat:"
    #     print ic.get_ir_to_rgb_rmat()
    #     print "tvec:"
    #     print ic.get_ir_to_rgb_tvec()
    #     rate = rospy.Rate(10)
    #     while not rospy.is_shutdown():
    #     	ic.register_depth_to_rgb()
    #     	rate.sleep()
    # except KeyboardInterrupt:
    #     print("Shutting down")
    cv2.destroyAllWindows()
