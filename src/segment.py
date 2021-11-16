#! /usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2, Image
import ros_numpy
import message_filters as msgf
import cv_bridge



class Segmenter():
    def __init__(self):
        self.publisher = rospy.Publisher("road_points",PointCloud2,queue_size=100)
        self.model

    def init_model():

    def pc_cb(self,pc_data,image):
        array = ros_numpy.point_cloud2.pointcloud2_to_array(pc_data)
        
        # Pass through model
        # Preprocess Image
        
        model = DeepLabv3()
        model.load_weights()
        op = model.predict(image)

        # Publish PCloud

        # Convert intensity values to 255 scale?

        # Publish
        array['intensity'] = op
        msg = ros_numpy.point_cloud2.array_to_pointcloud2(array)
        self.publisher.publish(msg)


def main():
    rospy.init_node("Road_Segment",anonymous=True)

    pc_sub = rospy.Subscriber("/os_cloud_node/points",PointCloud2,pc_cb)
    pc_sub = msgf.Subscriber("/os_cloud_node/points")
    img_sub = msgf/msgf.Subscriber("/img_node/signal_image")


    while(not rospy.is_shutdown()):
        rospy.spin()

# Left
# Add time_synchronizer
# Define model in Class,load weights
# Preprocess ip image

