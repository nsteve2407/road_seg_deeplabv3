
# Road_Seg_Deeplabv3

Road surface segmentation in 3D pointclouds can be computationally intensive as this may require 3D convolutions. On the other hand more classical approaches to segment road surfaces typically focus on identifying features to differentiate between road and non- road points. This approach usually requires proper identification of features as well as their threshold values.
The current work focuses on leveraging the feature learning abilities  of Convolutional Neural Networks on 2D images to segment the road in 3D pointclouds, by treating the pointcloud as a range image. Initial results show an accuracy of ~94% as well as reasonable runtime of 10Hz.
The trained model has been packaged as a ROS node for further use.


https://user-images.githubusercontent.com/91099619/143667401-f051cf09-a954-4ea5-b054-5c844688aef1.mp4




### Architecture
The repo makes use of [DeepLabv3 plus](https://arxiv.org/abs/1802.02611) - a state of the art 2D image segmentation model pretrained on the [Imagenet Dataset](https://image-net.org/)

The model was trained on 200 annotated LiDAR images collected using an Ouster LiDAR
### File Structure
-> launch : launch file for ROS node
-> src : source code for ROS node.
-> weights: contains trained model weights
### Dependencies
- Python3 
- ROS Noetic (recommended)
- Tensorflow2 ( GPU support recommended for training or realtime inference)

### Installation
 Clone the repo into your catkin workspace:
 > git clone
 
Check the path to weights file on line 57 in segment.py

After making the workspace, run
> roslaunch road_seg_deeplabv3 segment.launch

To train the model use 'deeplabv3_plus.ipynb' The images and masks must be placed in a folder called data inside the main directory.

