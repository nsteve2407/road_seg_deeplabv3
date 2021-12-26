import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time


infile = "./rellis.avi"
outfile = "rellis_segmented.avi"

reader = cv2.VideoCapture(infile)
writer  = cv2.VideoWriter(outfile,cv2.VideoWriter_fourcc('M','J','P','G'),5,(1024,128))
weights_file  ="./checkpoints/aug_model_free/acc945.h5"

if reader.isOpened():
    print("Video file loaded !")

# define model

# blocks --------------------------------


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


# Model ----------------------------------------------------------------------

IMAGE_SIZE = 128
NUM_CLASSES = 1

def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, 1024, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    resnet50.trainable = True
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], 1024 // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], 1024 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same",activation='sigmoid')(x)
    return keras.Model(inputs=model_input, outputs=model_output)

model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
model.load_weights(weights_file)

print("Model Weights Loaded\n")
# --------------------------------------Functions for Inference ----------------------------------------------

def infer(model, image_tensor):
    # image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)
    image_tensor = tf.convert_to_tensor(image_tensor)
    image_tensor.set_shape([None, None, 3])
    image_tensor = tf.image.resize(images=image_tensor, size=[128, 1024])
    image_tensor = image_tensor /255

    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    # predictions = np.argmax(predictions,axis=1)
    predictions = np.round(predictions)
    predictions = np.expand_dims(predictions,axis=-1)
    return predictions


def decode_segmentation_masks(mask, colormap, n_classes):
    # r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.float32)
    b = np.zeros_like(mask).astype(np.float32)
    # for l in range(0, n_classes):
    #     l+=1
    #     idx = mask == l
    #     r[idx] = colormap[l, 0]
    #     # g[idx] = colormap[l, 1]
    #     # b[idx] = colormap[l, 2]
    r = mask
    rgb = np.stack([b, g, r],axis=-1)
    rgb = np.reshape(rgb,(128,1024,3))
    # print('Source mask shape:{}'.format(rgb.shape))
    return rgb


def get_overlay(image, colored_mask):
    # image = tf.keras.preprocessing.image.array_to_img(image)
    # image = np.array(image).astype(np.float32)
    np.reshape(colored_mask,(128,1024,3))
    # print('Mask:{}'.format(colored_mask.shape))
    # print('Image:{}'.format(image.shape))
    overlay = cv2.addWeighted(image, 0.8, colored_mask.astype(np.uint8), 70.0, 0)
    return overlay


# Run Inference ====================================================
print("Running inference:\n")

# Warm up---------------------------------------------------
got_frame, frame = reader.read()
for i in range(100):
    predictions  = infer(model,frame)


# -------------------

time_total = 0
count = 0

while reader.isOpened():

    got_frame, frame = reader.read()

    if got_frame:
        start = time.time()
        predictions  = infer(model,frame)
        time_total += time.time() - start
        count +=1
        rgb_mask = decode_segmentation_masks(predictions,[],NUM_CLASSES)
        segmented_image = get_overlay(frame,rgb_mask)

        writer.write(segmented_image)
    else:
        break

print("Inference Comlete, File saved as {}!".format(outfile))

print(" Average inference time:{}".format(time_total/count))

reader.release()
writer.release()


