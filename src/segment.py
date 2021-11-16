#! /usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2, Image
import ros_numpy
import message_filters as msgf
import cv_bridge
import tensorflow as tf
import numpy as np
from deeplabv3 import DeeplabV3Plus


class Segmenter():
    def __init__(self,path_to_weights):
        self.publisher = rospy.Publisher("road_points",PointCloud2,queue_size=100)
        self.model = []
        self.num_classes = 1
        self.img_height = 128
        self.path = path_to_weights

    def init_model(self):
        self.model = DeeplabV3Plus(self.img_height,self.num_classes)
        self.model.load_weights(self.path)

    def pc_cb(self,pc_data,image_msg):
        array = ros_numpy.point_cloud2.pointcloud2_to_array(pc_data)
        bridge = cv_bridge.CvBridge()
        img = bridge.imgmsg_to_cv2(image_msg)

        image_tensor = tf.convert_to_tensor(img)
        image_tensor.set_shape([None, None, 3])
        image_tensor = tf.image.resize(images=image_tensor, size=[128, 1024])
        image_tensor = image_tensor /255

        op = self.model.predict(np.expand_dims((image_tensor), axis=0))
        op  = np.squeeze(op)
        op = np.round(op)
        op = np.expand_dims(op,axis=-1)

        # Publish PCloud

        # Convert intensity values to 255 scale?
        op = op*255

        # Publish
        array['intensity'] = op # Need to change dtpye?
        msg = ros_numpy.point_cloud2.array_to_pointcloud2(array)
        self.publisher.publish(msg)


def main():
    rospy.init_node("Road_Segment",anonymous=True)
    weights_file  = " "
    segmenter = Segmenter(weights_file)

    pc_sub = msgf.Subscriber("/os_cloud_node/points",PointCloud2)
    img_sub = msgf.Subscriber("/img_node/signal_image",Image)

    ats = msgf.ApproximateTimeSynchronizer([pc_sub,img_sub],10,0.00001)

    ats.registerCallback(segmenter.pc_cb)



    while(not rospy.is_shutdown()):
        rospy.spin()

# Left
# Add time_synchronizer
# Define model in Class,load weights
# Preprocess ip image

