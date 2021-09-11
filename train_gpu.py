import tensorflow as tf
import os
import json
import glob
from model import google_net
from tqdm import tqdm
from tensorflow.keras import *


def main():
    # 确认使用的gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        gpu0 = gpus[0]  # 如果有多个GPU，仅使用第0个GPU
        tf.config.experimental.set_memory_growth(gpu0, True)  # 设置GPU显存用量按需使用
        # 或者也可以设置GPU显存为固定使用量(例如：4G)
        # tf.config.experimental.set_virtual_device_configuration(gpu0,
        # [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        tf.config.set_visible_devices([gpu0], "GPU")

    # 预处理方法
    def pre_treatment(filename, label):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [target_height, target_width])
        image = (image / 255 - 0.5) * 2
        label = tf.one_hot(label, depth=len(data_cls))
        return image, label

    # 必要参数
    train_dir = r"C:\Users\lzy99\Pictures\Saved Pictures\train"
    validation_dir = r"C:\Users\lzy99\Pictures\Saved Pictures\val"
    epochs = 10
    batch_size = 5
    target_height = 224
    target_width = 224

    # 直接使用字典序号作为标签
    # 并将其写入json暂存 供解读预测结果使用
    data_cls = [cla for cla in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, cla))]
    class_dict = dict((value, index) for index, value in enumerate(data_cls))
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json.dumps(class_dict, indent=4))

    # 构建训练集
    train_image_list = glob.glob(train_dir + "/*/*.jpg")
    label_list = [class_dict[path.split(os.path.sep)[-2]] for path in train_image_list]
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, label_list)) \
        .map(pre_treatment) \
        .batch(batch_size) \
        .shuffle(buffer_size=1000)\
        .prefetch(tf.data.experimental.AUTOTUNE).cache()
    # 构建测试集
    test_image_list = glob.glob(validation_dir + "/*/*.jpg")
    test_dataset = tf.data.Dataset.from_tensor_slices((test_image_list, label_list)) \
        .map(pre_treatment) \
        .batch(batch_size) \
        .shuffle(buffer_size=1000) \
        .prefetch(tf.data.experimental.AUTOTUNE).cache()

    # 实例化模型
    model = google_net(im_height=224, im_width=224, class_num=len(data_cls), is_train=True)
    model.summary()
    # 加载已保存模型
    checkpoint_save_path = "./save_weights/myGoogLeNet.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('----load model from disk----')
        model.load_weights(checkpoint_save_path)
    # 定义损失函数和优化器
    loss_object = losses.CategoricalCrossentropy(from_logits=False)
    optimizer = optimizers.Adam(learning_rate=0.0003)

    train_loss = metrics.Mean(name='train_loss')
    train_accuracy = metrics.CategoricalAccuracy(name='train_accuracy')

    test_loss = metrics.Mean(name='test_loss')
    test_accuracy = metrics.CategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            aux1, aux2, output = model(images, training=True)
            loss1 = loss_object(labels, aux1)
            loss2 = loss_object(labels, aux2)
            loss3 = loss_object(labels, output)
            loss = loss1 * 0.3 + loss2 * 0.3 + loss3
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss.update_state(loss)
        train_accuracy.update_state(labels, output)

    @tf.function
    def test_step(images, labels):
        _, _, output = model(images, training=False)
        t_loss = loss_object(labels, output)

        test_loss.update_state(t_loss)
        test_accuracy.update_state(labels, output)

    def train_model(train_ds, val_ds, epochs):
        best_loss = float("inf")
        for epoch in tf.range(1, epochs + 1):

            for features, labels in train_ds:
                train_step(features, labels)

            for features, labels in val_ds:
                test_step(features, labels)

            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print(template.format(epoch,
                                  train_loss.result(),
                                  train_accuracy.result() * 100,
                                  test_loss.result(),
                                  test_accuracy.result() * 100))

            if test_loss.result() < best_loss:
                best_loss = test_loss.result()
                model.save_weights("./save_weights/myGoogLeNet.ckpt".format(epoch), save_format='tf')

            train_loss.reset_states()
            test_loss.reset_states()
            train_accuracy.reset_states()
            test_accuracy.reset_states()

    train_model(train_dataset, test_dataset, epochs=epochs)


if __name__ == '__main__':
    main()
