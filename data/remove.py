import os

mask_path = './masks/cropped'
mlayer_path  = "./m_layer/cropped"
slayer_path = "./s_layer/cropped"


# cwd = os.curdir()
imgs = os.listdir(slayer_path)
masks = os.listdir(mask_path)

img_ = [os.path.splitext(x)[0] for x in imgs]
masks_ = [os.path.splitext(x)[0] for x in masks]


for file in img_:
    if file in masks_:
        pass
    else:
        os.remove(os.path.join(slayer_path,file+'.jpg'))








