import os
import glob
import json
import sys
import tensorflow as tf
import numpy as np

from model import google_net


def predict():
    # 读取图片
    img_path = r"C:\Users\lzy99\Pictures\Saved Pictures\train\dog\dog.0.jpg"
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [224, 224])
    image = (image / 255 - 0.5) * 2

    # 添加维度生成输入列表
    img = np.expand_dims(image, 0)

    # 从json读取模型参数
    json_path = './class_indices.json'
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    class_num = len(class_indict)

    # 重建模型
    weights_path = "./save_weights/myGoogLeNet.ckpt"
    if (len(glob.glob(weights_path + "*"))) != 2:
        print("weights file not found")
        sys.exit(1)
    model = google_net(class_num=class_num, is_train=False)
    model.load_weights(weights_path)

    # 取得预测结果
    result = np.squeeze(model.predict(img))
    predict_class = np.argmax(result)
    for key, value in class_indict.items():
        if value == predict_class:
            print("it might be : {} \nprob: {}".format(key, result[predict_class]))


if __name__ == "__main__":
    predict()