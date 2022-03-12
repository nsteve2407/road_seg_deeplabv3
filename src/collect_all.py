#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import os
import cv2
import cv_bridge as cvb
import numpy as np
import message_filters as mf

path_single_layer = "/home/mkz/catkin_ws/src/road_seg_deeplabv3/data/sample/"
path_multilayer = "/home/mkz/catkin_ws/src/road_seg_deeplabv3/data/sample/"
topic1 = '/img_node/signal_image'
topic2 = '/img_node/range_image'
topic3 = '/img_node/reflec_image'
# topic4 = '/img_node/nearir_image'
# count=0
def cb(i1,i2,i3):

    ip = input("\nPress enter to save the image:\n")
    if(ip == ''):
        B  = cvb.CvBridge()
        i1m = B.imgmsg_to_cv2(i1, desired_encoding='mono8')
        i2m = B.imgmsg_to_cv2(i2, desired_encoding='mono8')
        i3m = B.imgmsg_to_cv2(i3, desired_encoding='mono8')
        # i4m = B.imgmsg_to_cv2(i4, desired_encoding='mono8')
        fname = str(i1.header.stamp)+'.jpg'

        mlayer_image = np.dstack((i1m,i2m,i3m))

        cv2.imwrite(os.path.join(path_single_layer,fname),i1m)
        cv2.imwrite(os.path.join(path_multilayer,'m'+fname),mlayer_image)


rospy.init_node("Image_saver",anonymous=True)

sub1 = mf.Subscriber(topic1,Image)
sub2 = mf.Subscriber(topic2,Image)
sub3 = mf.Subscriber(topic3,Image)
# sub4 = mf.Subscriber(topic4,Image)


ats = mf.ApproximateTimeSynchronizer([sub1,sub2,sub3],10,0.1)
ats.registerCallback(cb)

rospy.spin()






