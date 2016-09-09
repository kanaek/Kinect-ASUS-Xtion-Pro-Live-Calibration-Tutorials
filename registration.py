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

width = 0.0348
delta_x = 0   #0.0824
delta_y = 0   #0.064
x_num = 8
y_num = 6
objpoints = np.zeros((x_num * y_num, 3), np.float32)

class registration:
    def __init__(self):

        self.br = CvBridge()

        # If you subscribe /camera/depth_registered/hw_registered/image_rect topic, the depth image and rgb image are 
        # already registered. So you don't need to call register_depth_to_rgb()
        # self.depth_image_sub = rospy.Subscriber("/camera/depth_registered/hw_registered/image_rect",Image,self.depth_callback)

        self.depth_image_sub = rospy.Subscriber("/camera/depth/image_rect",Image,self.depth_callback)
        self.rgb_image_sub = rospy.Subscriber("/camera/rgb/image_rect_color",Image,self.rgb_callback)
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

        self.load_extrinsics()
        self.load_intrinsics()
        self.depth_image = None
        self.rgb_image = None
        self.count = 0
        # self.depth_image = cv2.imread("/home/chentao/depth.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)
        # self.rgb_image = cv2.imread("/home/chentao/rgb.png")
    
    def depth_callback(self,data):
    	try:
    		self.depth_image= self.br.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)
        # print "depth"


    def rgb_callback(self,data):
        try:
        	self.rgb_image = self.br.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # print "rgb"


    def load_extrinsics(self):
       ir_stream = open("/home/chentao/kinect_calibration/ir_camera_pose.yaml", "r")
       ir_doc = yaml.load(ir_stream)
       self.ir_rmat = np.array(ir_doc['rmat']).reshape(3,3)
       self.ir_tvec = np.array(ir_doc['tvec'])
       ir_stream.close()

       rgb_stream = open("/home/chentao/kinect_calibration/rgb_camera_pose.yaml", "r")
       rgb_doc = yaml.load(rgb_stream)
       self.rgb_rmat = np.array(rgb_doc['rmat']).reshape(3,3)
       self.rgb_tvec = np.array(rgb_doc['tvec'])
       rgb_stream.close()

    def load_intrinsics(self):
    	depth_stream = open("/home/chentao/kinect_calibration/depth_1504270110.yaml", "r")
        depth_doc = yaml.load(depth_stream)
        self.depth_mtx = np.array(depth_doc['camera_matrix']['data']).reshape(3,3)
        self.depth_dist = np.array(depth_doc['distortion_coefficients']['data'])
        depth_stream.close()

        rgb_stream = open("/home/chentao/kinect_calibration/rgb_1504270110.yaml", "r")
        rgb_doc = yaml.load(rgb_stream)
        self.rgb_mtx = np.array(rgb_doc['camera_matrix']['data']).reshape(3,3)
        self.rgb_dist = np.array(rgb_doc['distortion_coefficients']['data'])
        rgb_stream.close()


    def ir_to_rgb(self):
        if self.rgb_rmat != None and self.rgb_rmat != None and self.ir_rmat != None and self.ir_tvec != None:
            self.ir_to_rgb_rmat = np.dot(self.rgb_rmat, np.transpose(self.ir_rmat))
            self.ir_to_rgb_tvec = self.rgb_tvec - np.dot(self.ir_to_rgb_rmat, self.ir_tvec)

    def register_depth_to_rgb(self):
        if self.depth_image == None or self.rgb_image == None:
            return

        self.registered_rgb = np.zeros(self.rgb_image.shape)


        depth_pix_point = np.ones((self.depth_image.shape[0] * self.depth_image.shape[1],3))
        index = np.transpose(np.mgrid[0:self.depth_image.shape[1], 0:self.depth_image.shape[0]]).reshape(-1,2)
        depth_pix_point[:,:2] = index.copy()

        depth_pix_point = (depth_pix_point.T * self.depth_image[index[:,1].astype(int), index[:,0].astype(int)].reshape(1,-1)).T     #each row is a point

        depth_coord_point = np.dot(np.linalg.inv(self.depth_mtx), depth_pix_point.T)      #each column is a point

        rgb_coord_point = np.dot(self.ir_to_rgb_rmat, depth_coord_point).reshape(3,-1) + self.ir_to_rgb_tvec     # each column is a point

        rgb_pix_point = np.dot(self.rgb_mtx, rgb_coord_point)    #each column is a point

        u = rgb_pix_point[0,:].astype(float)
        v = rgb_pix_point[1,:].astype(float)
        w = rgb_pix_point[2,:].astype(float)
        rgb_x = (u / w).astype(int)     # row vector
        rgb_y = (v / w).astype(int)     # row vector
        rgb_x = np.clip(rgb_x, 0, self.rgb_image.shape[1] - 1)
        rgb_y = np.clip(rgb_y, 0, self.rgb_image.shape[0] - 1)



        self.registered_rgb[index[:,1].astype(int), index[:,0].astype(int), :] = self.rgb_image[rgb_y, rgb_x, :]
        self.registered_rgb = self.registered_rgb.astype(int)

        depth_min = np.nanmin(self.depth_image)
        depth_max = np.nanmax(self.depth_image)


        depth_img = self.depth_image.copy()
        depth_img[np.isnan(self.depth_image)] = depth_min
        depth_img = ((depth_img - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        
        filename1 = '/home/chentao/figures/regi' + str(self.count) + '.png'
        filename2 = '/home/chentao/figures/depth' + str(self.count) + '.png'
        filename3 = '/home/chentao/figures/rgb' + str(self.count) + '.png'
        self.count += 1

        

        cv2.imwrite(filename1,self.registered_rgb)   
        cv2.imwrite(filename2,depth_img)
        cv2.imwrite(filename3,self.rgb_image)
        


    def get_ir_to_rgb_rmat(self):
        if self.ir_to_rgb_rmat != None:
            return self.ir_to_rgb_rmat


    def get_ir_to_rgb_tvec(self):
        if self.ir_to_rgb_tvec != None:
            return self.ir_to_rgb_tvec


if __name__ == "__main__":
    for i in range(y_num):
        for j in range(x_num):
            index = i * x_num + j
            objpoints[index,0] = delta_x + j * width
            objpoints[index,1] = delta_y + (y_num - 1 - i) * width
            objpoints[index,2] = 0
    rospy.init_node('registration')
    ic = registration()
    try:
        ic.ir_to_rgb()
        print "=================================="
        print "rmat:"
        print ic.get_ir_to_rgb_rmat()
        print "tvec:"
        print ic.get_ir_to_rgb_tvec()
        rate = rospy.Rate(10)
        # ic.register_depth_to_rgb()
        while not rospy.is_shutdown():
            ic.register_depth_to_rgb()
            rate.sleep()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
