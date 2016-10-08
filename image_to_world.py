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


# The calibration result might have some constant offset
x_offset = 0.0
y_offset = -0.001
z_offset = 0.001



class map_img_to_world:
    def __init__(self):

        self.br = CvBridge()

        # If you subscribe /camera/depth_registered/hw_registered/image_rect topic, the depth image and rgb image are 
        # already registered. So you don't need to call register_depth_to_rgb()
        self.depth_image_sub = rospy.Subscriber("/camera/depth_registered/hw_registered/image_rect",Image,self.depth_callback)
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
        self.ir_to_world_tvec = None
        self.ir_to_rgb_rmat = None
        self.load_extrinsics()
        self.load_intrinsics()
        self.depth_image = None
        self.rgb_image = None
        self.count = 0
        self.drawing = False # true if mouse is pressed
        self.rect_done = False
        self.ix1 = -1
        self.iy1 = -1
        self.ix2 = -1
        self.iy2 = -1

        cv2.namedWindow('RGB Image')
        cv2.setMouseCallback('RGB Image',self.draw_rect)
    
    def depth_callback(self,data):
        try:
            self.depth_image= self.br.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)
        # print "depth"

        depth_min = np.nanmin(self.depth_image)
        depth_max = np.nanmax(self.depth_image)


        depth_img = self.depth_image.copy()
        depth_img[np.isnan(self.depth_image)] = depth_min
        depth_img = ((depth_img - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        cv2.imshow("Depth Image", depth_img)
        cv2.waitKey(5)
        # stream = open("/home/chentao/depth_test.yaml", "w")
        # data = {'img':depth_img.tolist()}
        # yaml.dump(data, stream)


    def rgb_callback(self,data):
        try:
            self.rgb_image = self.br.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        

        if self.drawing or self.rect_done:
            if (self.ix1 != -1 and self.iy1 != -1 and self.ix2 != -1 and self.iy2 != -1):
                cv2.rectangle(self.rgb_image,(self.ix1,self.iy1),(self.ix2,self.iy2),(0,255,0),2)
                if self.rect_done:
                    center_point = self.get_center_point()
                    cv2.circle(self.rgb_image, tuple(center_point.astype(int)), 3, (0,0,255),-1)
        
        cv2.imshow('RGB Image', self.rgb_image)
        cv2.waitKey(5)
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

       self.rgb_to_world_rmat = self.rgb_rmat.T
       self.rgb_to_world_tvec = -np.dot(self.rgb_rmat.T, self.rgb_tvec)

       self.ir_to_world_rmat = self.ir_rmat.T
       self.ir_to_world_tvec = -np.dot(self.ir_rmat.T, self.ir_tvec)


    def load_intrinsics(self):
    	depth_stream = open("/home/chentao/kinect_calibration/depth_0000000000000000.yaml", "r")
        depth_doc = yaml.load(depth_stream)
        self.depth_mtx = np.array(depth_doc['camera_matrix']['data']).reshape(3,3)
        self.depth_dist = np.array(depth_doc['distortion_coefficients']['data'])
        depth_stream.close()

        rgb_stream = open("/home/chentao/kinect_calibration/rgb_0000000000000000.yaml", "r")
        rgb_doc = yaml.load(rgb_stream)
        self.rgb_mtx = np.array(rgb_doc['camera_matrix']['data']).reshape(3,3)
        self.rgb_dist = np.array(rgb_doc['distortion_coefficients']['data'])
        rgb_stream.close()

    def img_to_world(self, pix_point):
        if self.depth_image == None or self.rgb_image == None:
            return

        # # pix_point is (u,v) : the coordinates on the image
        # depth_pix_point = np.array([pix_point[0], pix_point[1], 1]) * self.depth_image[pix_point[1], pix_point[0]]
        # depth_coord_point = np.dot(np.linalg.inv(self.depth_mtx), depth_pix_point.reshape(-1,1))
        # point_in_world = np.dot(self.ir_to_world_rmat, depth_coord_point.reshape(-1,1)) + self.ir_to_world_tvec
        # point_in_world[0] += x_offset
        # point_in_world[1] += y_offset
        # point_in_world[2] += z_offset

        depth_pix_point = np.array([pix_point[0], pix_point[1], 1]) * self.depth_image[pix_point[1], pix_point[0]]
        depth_coord_point = np.dot(np.linalg.inv(self.rgb_mtx), depth_pix_point.reshape(-1,1))

        point_in_world = np.dot(self.rgb_to_world_rmat, depth_coord_point.reshape(-1,1)) + self.rgb_to_world_tvec
        point_in_world[0] += x_offset
        point_in_world[1] += y_offset
        point_in_world[2] += z_offset
        return point_in_world

    def get_center_point(self):
        if (self.ix1 != -1 and self.iy1 != -1 and self.ix2 != -1 and self.iy2 != -1):
            pix_point = np.zeros(2)
            pix_point[0] = (self.ix1 + self.ix2) / 2
            pix_point[1] = (self.iy1 + self.iy2) / 2
            # print "center point in image: ",pix_point
            return pix_point

    def draw_rect(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.rect_done = False
            self.ix1 = x
            self.iy1 = y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                self.ix2 = x
                self.iy2 = y


        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.ix2 = x
            self.iy2 = y
            cv2.rectangle(self.rgb_image,(self.ix1,self.iy1),(self.ix2,self.iy2),(0,255,0),2)
            center_point = self.get_center_point()
            cv2.circle(self.rgb_image, tuple(center_point.astype(int)), 3, (0,0,255),-1)
            cv2.imshow('RGB Image', self.rgb_image)
            cv2.waitKey(5)
            self.rect_done = True


if __name__ == "__main__":
    rospy.init_node('map_img_to_world')
    ic = map_img_to_world()
    try:
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if ic.rect_done:
                point = ic.img_to_world(ic.get_center_point())
                if point != None:
                    print "================================="
                    print "center point in world coordinate:"
                    print point

            rate.sleep()

    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
