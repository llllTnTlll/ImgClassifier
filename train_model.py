import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 预处理方法
def pre_treatment(img):
    img = np.array(img).astype(np.float32)
    img = img / 255.
    img = (img - 0.5) * 2.0
    return img


# 必要参数
dir_path = './Data/train_set'
img_path = []                  # 保存训练照片的地址集
class_num = 0
epoch = 10
batch_size = 32

# 获取文件列表
folder_list = os.listdir(dir_path)
for folder in folder_list:
    class_num += 1
    img_path.append(os.path.sep.join([dir_path, folder]))

# 定义数据生成器
train_image_generator = ImageDataGenerator(preprocessing_function=pre_treatment,
                                           horizontal_flip=True)
validation_image_generator = ImageDataGenerator(preprocessing_function=pre_treatment)


