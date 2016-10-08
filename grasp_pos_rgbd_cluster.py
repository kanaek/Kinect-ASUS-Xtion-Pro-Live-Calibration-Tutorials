#!/usr/bin/env python


import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import linear_model
import numpy as np
from scipy import optimize
import yaml
import sys
from matplotlib import pyplot as plt
from scipy import stats
from center_and_normal.msg import center_and_normal




# The calibration result might have some constant offset
x_offset = 0.0
y_offset = -0.001
z_offset = 0.001



class grasp:
    def __init__(self):

        self.br = CvBridge()

        # If you subscribe /camera/depth_registered/hw_registered/image_rect topic, the depth image and rgb image are 
        # already registered. So you don't need to call register_depth_to_rgb()
        self.depth_image_sub = rospy.Subscriber("/camera/depth_registered/hw_registered/image_rect",Image,self.depth_callback)
        self.rgb_image_sub = rospy.Subscriber("/camera/rgb/image_rect_color",Image,self.rgb_callback)
        
        # self.pub = rospy.Publisher('chatter', String, queue_size=10)
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

        
        tempimg = self.rgb_image.copy()
        if self.drawing or self.rect_done:
            if (self.ix1 != -1 and self.iy1 != -1 and self.ix2 != -1 and self.iy2 != -1):
                cv2.rectangle(tempimg,(self.ix1,self.iy1),(self.ix2,self.iy2),(0,255,0),2)
                if self.rect_done:
                    center_point = self.get_center_point()
                    cv2.circle(tempimg, tuple(center_point.astype(int)), 3, (0,0,255),-1)
        
        cv2.imshow('RGB Image', tempimg)
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

        # pix_point is (u,v) : the coordinates on the image
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

 
    def get_orientation(self):
        if (self.ix1 != -1 and self.iy1 != -1 and self.ix2 != -1 and self.iy2 != -1):
            if self.ix1 > self.ix2:
                temp = self.ix2
                self.ix2 = self.ix1
                self.ix1 = temp

            if self.iy1 > self.iy2:
                temp = self.iy2
                self.iy2 = self.iy1
                self.iy1 = temp

            roi_width = self.ix2 - self.ix1 + 1
            roi_height = self.iy2 - self.iy1 + 1
            roi = self.rgb_image[self.iy1:self.iy2 + 1, self.ix1:self.ix2 + 1, :].reshape(-1,3).astype(float)
            depth_column = self.depth_image[self.iy1:self.iy2 + 1, self.ix1:self.ix2 + 1].reshape(-1,1).astype(float)
            invalid_indices = np.isnan(depth_column)
            depth_column[invalid_indices] = 0
            roi = np.hstack((roi, depth_column))
            roi = preprocessing.scale(roi)


            # # KMeans
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(roi)
            y = kmeans.predict(roi)
            y = y.reshape(roi_height, roi_width)


            # Find the mode of the cluster index in the boundary, assume that the mode represent the cluster of background
            wid = 10
            t = np.append(y[:,:wid].reshape(-1,1),y[:,y.shape[1] - wid:].reshape(-1,1))
            t = np.append(t, y[y.shape[0] - wid:,wid:y.shape[1] - wid].reshape(-1,1))
            t = np.append(t, y[0:wid, wid:y.shape[1] - wid].reshape(-1,1))

            # since the cluster index can only be 0 or 1 here,so if the background is 0, then our target is 1, vice versa.
            interested_cluster = 1 - stats.mode(t)[0][0]
            interested_cluster_indices = np.array(np.where(y == interested_cluster))

            interested_cluster_indices[0] += self.iy1
            interested_cluster_indices[1] += self.ix1

            tempimg = self.rgb_image.copy()
            tempimg[interested_cluster_indices[0], interested_cluster_indices[1],:] = np.zeros((1,3))


            # Grab Cut
            # mask = np.zeros(self.rgb_image.shape[:2],np.uint8)
 
            # bgdModel = np.zeros((1,65),np.float64)
            # fgdModel = np.zeros((1,65),np.float64)

            # ix1 = max(self.ix1,1)
            # iy1 = max(self.iy1,1)
            # ix2 = max(self.ix2,1)
            # iy2 = max(self.iy2,1)
            # ix1 = min(self.ix1,self.rgb_image.shape[1])
            # iy1 = min(self.iy1,self.rgb_image.shape[0])
            # ix2 = min(self.ix2,self.rgb_image.shape[1])
            # iy2 = min(self.iy2,self.rgb_image.shape[0])
            # # print "ix1: ",ix1
            # # print "iy1: ",iy1
            # # print "ix2: ",ix2
            # # print "iy2: ",iy2
            # rect = (ix1,iy1,ix2,iy2)
            # print "Grab Cut Started..."
            # cv2.grabCut(self.rgb_image,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)
            # print "Grab Cut done..."
            # # all 0-pixels and 2-pixels are put to 0 (ie background) and all 1-pixels and 3-pixels are put to 1(ie foreground pixels)
            # mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
            # interested_cluster_indices = np.where(mask2 == 1)
            # tempimg = self.rgb_image*mask2[:,:,np.newaxis]
            
            cv2.imshow("Segmentation",tempimg)
            cv2.waitKey(5)
            roi_points = []

            for i,j in zip(interested_cluster_indices[0], interested_cluster_indices[1]):
                pix_point =np.zeros(2)
                pix_point[0] = j
                pix_point[1] = i
                point_temp = self.img_to_world(pix_point)
                if np.isnan(point_temp).sum() == 0:
                    roi_points.append(point_temp)


            roi_points = np.array(roi_points).reshape(-1,3)
            # Remove the possible outliers

            # roi_points = roi_points[roi_points[:,2] > np.percentile(roi_points[:,2],25) and roi_points[:,2] < np.percentile(roi_points[:,2],75)]
            # print roi_points.shape
            roi_points = roi_points[roi_points[:,2] > np.percentile(roi_points[:,2],25)]
            # print roi_points.shape
            roi_points = roi_points[roi_points[:,2] < np.percentile(roi_points[:,2],50 / 75.0 * 100)]
            # print roi_points.shape

            # Find Normal Vector, use a plane to fit these data
            y = roi_points[:,2]
            X = roi_points[:,:2]
           
            model = linear_model.LinearRegression()
            model.fit(X, y) 
            normal_vector = np.zeros(3)
            normal_vector[0] = -model.coef_[0]
            normal_vector[1] = -model.coef_[1]
            normal_vector[2] = 1

            cos_alpha = np.zeros(3)
            alpha = np.zeros(3)
            for i in range(3):
                cos_alpha[i] = normal_vector[i] / np.linalg.norm(normal_vector)
            alpha = np.arccos(cos_alpha)

            # # print "normal vector:",normal_vector
            # # print "cos_alpha:",cos_alpha
            # # print "alpha:",alpha


            # # Find the radius for the cylinder
            # # https://scipy.github.io/old-wiki/pages/Cookbook/Least_Squares_Circle.html


            # self.points_x = X[:,0]
            # self.points_y = X[:,1]
            # x_m = np.mean(self.points_x)
            # y_m = np.mean(self.points_y)
            # center_estimate = np.array([x_m, y_m])
            # center, ier = optimize.leastsq(self.circle_fit_cost, center_estimate)
            # R = np.sqrt((self.points_x - center[0]) ** 2 + (self.points_y - center[1]) ** 2).mean()

            # return alpha, R

            return alpha, normal_vector


    def circle_fit_cost(self, c):
        Ri = np.sqrt((self.points_x - c[0]) ** 2 + (self.points_y - c[1]) ** 2)
        return Ri - Ri.mean()

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

            tempimg = self.rgb_image.copy()
            cv2.rectangle(tempimg,(self.ix1,self.iy1),(self.ix2,self.iy2),(0,255,0),2)
            center_point = self.get_center_point()

            cv2.circle(tempimg, tuple(center_point.astype(int)), 3, (0,0,255),-1)
            cv2.imshow('RGB Image', tempimg)
            cv2.waitKey(5)
            self.rect_done = True


if __name__ == "__main__":
    rospy.init_node('center_and_normal_with_rgbd')
    ic = grasp()
    center_and_normal_pub = rospy.Publisher('center_and_normal', center_and_normal,queue_size=10)
    try:
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if ic.rect_done:
                point = ic.img_to_world(ic.get_center_point())
                # orientation, R = ic.get_orientation()
                orientation, normal_vector = ic.get_orientation()
                if point != None and orientation != None:
                    print "================================="
                    print "center point in world coordinate system:"
                    print point.reshape(3)
                    print "Normal vector of the plane with respect to world coordinate axes(X, Y, Z):"
                    print normal_vector
                    # print "radius of the cylinder:",R

                    msg = center_and_normal()
                    temp = np.zeros(6)
                    temp[0:3] = point.reshape(1,3)
                    temp[3:6] = orientation.reshape(1,3)
                    msg.position = temp
                    center_and_normal_pub.publish(msg)

            rate.sleep()

    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
