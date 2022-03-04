#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, Image
import ros_numpy
import message_filters as msgf
import cv_bridge
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import cv2
def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)

def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, 352, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    resnet50.trainable = True
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], 352 // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], 352 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same",activation='sigmoid')(x)
    return keras.Model(inputs=model_input, outputs=model_output)



class Segmenter():
    def __init__(self,path_to_weights,publish_image=True,publish_tv=True):
        self.publisher = rospy.Publisher("road_points",PointCloud2,queue_size=100)
        self.num_classes = 1
        self.img_height = 288
        self.path = path_to_weights
        self.model = DeeplabV3Plus(self.img_height,self.num_classes)
        self.model.load_weights(self.path)
        self.img_publisher = rospy.Publisher("road_segment_image",Image,queue_size=100)
        self.tv_publisher  = rospy.Publisher("road_segment_image_top_view",Image,queue_size=100)
        self.publish_image = publish_image
        self.publish_top_view  = publish_tv
        self.TV_m = np.load("/home/mkz/catkin_ws/src/road_seg_deeplabv3/data/mono/TV_tf.npy")

    def decode_segmentation_masks(self,mask, colormap, n_classes):
        
        r = np.zeros_like(mask).astype(np.float32)
        b = np.zeros_like(mask).astype(np.float32)
        g = mask
        rgb = np.stack([b, g, r],axis=-1)
        rgb = np.reshape(rgb,(288,352,3))
        # print('Source mask shape:{}'.format(rgb.shape))
        return rgb


    def get_overlay(self,image, colored_mask):
        # np.reshape(colored_mask,(128,512,3))
        overlay = cv2.addWeighted(image, 0.8, colored_mask.numpy().astype(np.uint8), 70.0, 0)
        return overlay

    def pc_cb(self,pc_data,image_msg):
        crop =320
        array = ros_numpy.point_cloud2.pointcloud2_to_array(pc_data)
        bridge = cv_bridge.CvBridge()
        img = bridge.imgmsg_to_cv2(image_msg,desired_encoding="bgr8")
        img = img[:crop,:]
        img_c  = bridge.imgmsg_to_cv2(image_msg,desired_encoding="bgr8")
        img_c = img_c[:crop,:,:]
        image_tensor = tf.convert_to_tensor(img)

        image_tensor = tf.stack([image_tensor,image_tensor,image_tensor],axis=-1)
        image_tensor = tf.expand_dims(image_tensor,0)

        image_tensor = tf.image.resize(images=image_tensor, size=[288, 352])
        image_tensor = image_tensor /255

        op = self.model.predict(image_tensor)
        op = np.squeeze(op)

        # Crop /Uncrop
        # zero = np.zeros((128,256),dtype=op.dtype)
        # op_full = np.hstack([zero,op,zero])
        # op = np.squeeze(op_full)
        # op = np.round(op)
        # op = np.expand_dims(op,axis=-1)

        # Publish PCloud

        # Convert intensity values to 255 scale?
        op_rs = op*255
        op_rs = np.where(op_rs>127.0,255.0,0.0)

        # Publish
        pcd = np.copy(array)
        pcd['intensity'] = op_rs.astype('<f4') # Need to change dtpye?
        msg = ros_numpy.point_cloud2.array_to_pointcloud2(pcd)
        msg.header = pc_data.header
        msg.header.stamp = rospy.Time.now() #comment if realtime use
        self.publisher.publish(msg)

        if self.publish_image:
            op = np.expand_dims(op,axis=-1)
            rgb_mask = self.decode_segmentation_masks(op,[],self.num_classes)
            rgb_mask = tf.image.resize(images=rgb_mask, size=[crop, 640])
            # rgb_mask = tf.image.resize(images=rgb_mask, size=[img.shape[0],img.shape[1]])
            segmented_image = self.get_overlay(np.stack([img,img,img],axis=-1),rgb_mask)
            segmented_image = bridge.cv2_to_imgmsg(segmented_image,"bgr8")
            self.img_publisher.publish(segmented_image)
            
    def img_cb(self,image_msg):
        crop =320
        bridge = cv_bridge.CvBridge()
        img = bridge.imgmsg_to_cv2(image_msg,desired_encoding="mono8")
        img = img[:crop,:]
        img_c  = bridge.imgmsg_to_cv2(image_msg,desired_encoding="bgr8")
        img_c = img_c[:crop,:,:]
        image_tensor = tf.convert_to_tensor(img)

        image_tensor = tf.stack([image_tensor,image_tensor,image_tensor],axis=-1)
        image_tensor = tf.expand_dims(image_tensor,0)

        image_tensor = tf.image.resize(images=image_tensor, size=[288, 352])
        image_tensor = image_tensor /255

        op = self.model.predict(image_tensor)
        op = np.squeeze(op)
        op = np.expand_dims(op,axis=-1)        
        rgb_mask = self.decode_segmentation_masks(op,[],self.num_classes)
        rgb_mask = tf.image.resize(images=rgb_mask, size=[crop, 640])
        # rgb_mask = tf.image.resize(images=rgb_mask, size=[img.shape[0],img.shape[1]])
        segmented_image = self.get_overlay(np.stack([img,img,img],axis=-1),rgb_mask)
        segmented_image_msg = bridge.cv2_to_imgmsg(segmented_image,"bgr8")
        self.img_publisher.publish(segmented_image_msg)

        if self.publish_top_view:
            img_tv = cv2.warpPerspective(segmented_image,self.TV_m,dsize=(segmented_image.shape[1],segmented_image[0]))
            img_tv_msg = bridge.cv2_to_imgmsg(img_tv,encoding='bgr8')
            img_tv_msg.header.stamp = image_msg.header.stamp
            self.tv_publisher.publish(img_tv_msg)






def main():
    rospy.init_node("Road_Segment",anonymous=True)
    weights_file  = "/home/mkz/catkin_ws/src/road_seg_deeplabv3/weights/rtk_gs_model.h5"
    segmenter = Segmenter(weights_file)

    # pc_sub = msgf.Subscriber("/os_cloud_node/points",PointCloud2)
    # img_sub = msgf.Subscriber("/img_node/signal_image",Image)

    # ats = msgf.ApproximateTimeSynchronizer([pc_sub,img_sub],10,0.00001)

    # ats.registerCallback(segmenter.pc_cb)

    img_sub = rospy.Subscriber("/usb_cam/image_raw",Image,segmenter.img_cb)

    rospy.spin()





# Left
# Add time_synchronizer
# Define model in Class,load weights
# Preprocess ip image
if __name__ == "__main__":
    main()
