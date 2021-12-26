#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, Image
import ros_numpy
import message_filters as msgf
import cv_bridge
import tensorflow as tf
import numpy as np
from deeplab_s_layer import DeeplabV3Plus
import cv2


class Segmenter():
    def __init__(self,path_to_weights,publish_image=True):
        self.publisher = rospy.Publisher("road_points",PointCloud2,queue_size=100)
        self.num_classes = 1
        self.img_height = 128
        self.path = path_to_weights
        self.model = DeeplabV3Plus(self.img_height,self.num_classes)
        self.model.load_weights(self.path)
        self.img_publisher = rospy.Publisher("road_segment_image",Image,queue_size=100)
        self.publish_image = publish_image

    def decode_segmentation_masks(self,mask, colormap, n_classes):
        
        g = np.zeros_like(mask).astype(np.float32)
        b = np.zeros_like(mask).astype(np.float32)
        r = mask
        rgb = np.stack([b, g, r],axis=-1)
        rgb = np.reshape(rgb,(128,512,3))
        # print('Source mask shape:{}'.format(rgb.shape))
        return rgb


    def get_overlay(self,image, colored_mask):
        # np.reshape(colored_mask,(128,512,3))
        overlay = cv2.addWeighted(image, 0.8, colored_mask.astype(np.uint8), 70.0, 0)
        return overlay

    def pc_cb(self,pc_data,image_msg):
        array = ros_numpy.point_cloud2.pointcloud2_to_array(pc_data)
        bridge = cv_bridge.CvBridge()
        img = bridge.imgmsg_to_cv2(image_msg,desired_encoding="bgr8")
        img = img[:,256:768,:]
        image_tensor = tf.convert_to_tensor(img)
        image_tensor.set_shape([None, None, 3])
        image_tensor = tf.image.resize(images=image_tensor, size=[128, 512])
        image_tensor = image_tensor /255

        op = self.model.predict(np.expand_dims((image_tensor), axis=0))
        op  = np.squeeze(op)
        zero = np.zeros((128,256),dtype=op.dtype)
        op_full = np.hstack([zero,op,zero])
        op = np.squeeze(op_full)
        op = np.round(op)
        # op = np.expand_dims(op,axis=-1)

        # Publish PCloud

        # Convert intensity values to 255 scale?
        op_rs = op*255

        # Publish
        pcd = np.copy(array)
        pcd['intensity'] = op_rs.astype('<f4') # Need to change dtpye?
        msg = ros_numpy.point_cloud2.array_to_pointcloud2(pcd)
        msg.header = pc_data.header
        msg.header.stamp = rospy.Time.now() #comment if realtime use
        self.publisher.publish(msg)

        if self.publish_image:
            op = np.expand_dims(op,axis=-1)
            rgb_mask = self.decode_segmentation_masks(op[:,256:768],[],self.num_classes)
            segmented_image = self.get_overlay(img,rgb_mask)
            segmented_image = bridge.cv2_to_imgmsg(segmented_image,"bgr8")
            self.img_publisher.publish(segmented_image)




def main():
    rospy.init_node("Road_Segment",anonymous=True)
    weights_file  = "/home/mkz/catkin_ws/src/road_seg_deeplabv3/weights/s_layer-best.h5"
    segmenter = Segmenter(weights_file)

    pc_sub = msgf.Subscriber("/os_cloud_node/points",PointCloud2)
    img_sub = msgf.Subscriber("/img_node/signal_image",Image)

    ats = msgf.ApproximateTimeSynchronizer([pc_sub,img_sub],10,0.00001)

    ats.registerCallback(segmenter.pc_cb)

    rospy.spin()





# Left
# Add time_synchronizer
# Define model in Class,load weights
# Preprocess ip image
if __name__ == "__main__":
    main()
