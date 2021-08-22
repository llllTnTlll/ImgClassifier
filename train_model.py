import os
import model
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
train_dir = './Data/train_set'
validation_dir = './Data/validation_set'
img_path = []                  # 保存训练照片的地址集
class_num = 0
epoch = 10
batch_size = 32
target_height = 224
target_width = 224

# 获取文件列表
folder_list = os.listdir(train_dir)
for folder in folder_list:
    class_num += 1
    img_path.append(os.path.sep.join([train_dir, folder]))

# 定义训练集数据生成器
train_image_generator = ImageDataGenerator(preprocessing_function=pre_treatment,
                                           horizontal_flip=True,
                                           rotation_range=30,
                                           samplewise_center=True,
                                           )

train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           target_size=(target_height, target_width),
                                                           shuffle=True,
                                                           color_mode='rgb',
                                                           class_mode='categorical',
                                                           batch_size=batch_size,
                                                           )

# 定义测试集数据生成器
validation_image_generator = ImageDataGenerator(preprocessing_function=pre_treatment)

validation_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                     target_size=(target_height, target_width),
                                                                     shuffle=False,
                                                                     color_mode='rgb',
                                                                     class_mode='categorical',
                                                                     batch_size=batch_size,
                                                                     )

# train_num = train_data_gen.n    # 获取训练集图像数量

# 使用训练模式实例化GoogleNet模型
model = model.google_net(im_height=target_height, im_width=target_width, class_num=class_num, is_train=True)
model.summary()

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)    # 定义损失函数类型
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)                  # 定义优化器类型

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')


@tf.function
def train_step(imgs, labels):
    with tf.GradientTape() as tape:
        aux1, aux2, output = model(imgs, training=True)    # training为True启用Dropout
        loss1 = loss_object(labels, aux1)
        loss2 = loss_object(labels, aux2)
        loss3 = loss_object(labels, output)
        loss = loss1 * 0.3 + loss2 * 0.3 + loss3
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, output)


@tf.function
def test_step(imgs, labels):
    _, _, output = model(imgs, training=False)
    loss = loss_object(labels, output)

    val_loss(loss)
    val_accuracy(labels, output)

