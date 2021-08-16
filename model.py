from tensorflow.keras import layers, Sequential, models


class ConvBlk(layers.Layer):
    def __init__(self, kernal_siz: int, is_reduce: bool, filter_reduce: int, filter_nxn: int):
        super(ConvBlk, self).__init__()
        self.kernal_siz = kernal_siz

        if is_reduce is True:
            self.model = Sequential([
                layers.Conv2D(filters=filter_reduce, kernel_size=1, activation='relu'),
                layers.Conv2D(filters=filter_nxn, kernel_size=self.kernal_siz, activation='relu', padding='same')
            ])
        else:
            self.model = Sequential([
                layers.Conv2D(filters=filter_nxn, kernel_size=self.kernal_siz, activation='relu', padding='same')
            ])

    def call(self, inputs, **kwargs):
        branch_out = self.model(inputs)
        return branch_out


class InceptionBlk(layers.Layer):
    def __init__(self,
                 filter_1x1: int,
                 filter_3x3_reduce: int, filter_3x3: int,
                 filter_5x5_reduce: int, filter_5x5: int,
                 filter_pool_conv: int,
                 **kwargs):
        super(InceptionBlk, self).__init__(**kwargs)

        self.branch1 = ConvBlk(filter_reduce=0, filter_nxn=filter_1x1, kernal_siz=1, is_reduce=False)
        self.branch2 = ConvBlk(filter_reduce=filter_3x3_reduce, filter_nxn=filter_3x3, kernal_siz=3, is_reduce=True)
        self.branch3 = ConvBlk(filter_reduce=filter_5x5_reduce, filter_nxn=filter_5x5, kernal_siz=5, is_reduce=True)
        self.branch4 = Sequential([
            layers.MaxPool2D(pool_size=3, strides=1, padding='same'),
            layers.Conv2D(filters=filter_pool_conv, kernel_size=1, activation='relu')
        ])

    def call(self, inputs, **kwargs):
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        branch4 = self.branch4(inputs)
        inception_output = layers.concatenate([branch1, branch2, branch3, branch4], axis=3)
        return inception_output


class AuxiliaryBlk(layers.Layer):
    def __init__(self, cls_num: int, **kwargs):
        super(AuxiliaryBlk, self).__init__(**kwargs)
        self.cls_num = cls_num

        self.avg_pool = layers.AvgPool2D(pool_size=5, strides=3)
        self.conv = layers.Conv2D(128, kernel_size=1, activation='relu')
        self.d1 = layers.Dense(1024, activation='relu')
        self.d2 = layers.Dense(cls_num)
        self.softmax = layers.Softmax()

    def call(self, inputs, *args, **kwargs):
        aux_output = self.avg_pool(inputs)

        aux_output = self.conv(aux_output)
        aux_output = layers.Flatten()(aux_output)
        aux_output = layers.Dropout(rate=0.7)(aux_output)

        aux_output = self.d1(aux_output)
        aux_output = layers.Dropout(rate=0.7)(aux_output)

        aux_output = self.d2(aux_output)

        aux_output = self.softmax(aux_output)

        return aux_output


def google_net(is_train: bool, class_num, im_height=224, im_width=224):
    aux1 = None
    aux2 = None
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    # (None, 224, 224, 3)
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding="SAME", activation="relu", name="conv2d_1")(input_image)
    # (None, 112, 112, 64)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_1")(x)
    # (None, 56, 56, 64)
    x = layers.Conv2D(64, kernel_size=1, activation="relu", name="conv2d_2")(x)
    # (None, 56, 56, 64)
    x = layers.Conv2D(192, kernel_size=3, padding="SAME", activation="relu", name="conv2d_3")(x)
    # (None, 56, 56, 192)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_2")(x)

    # (None, 28, 28, 192)
    x = InceptionBlk(64, 96, 128, 16, 32, 32, name="inception_3a")(x)
    # (None, 28, 28, 256)
    x = InceptionBlk(128, 128, 192, 32, 96, 64, name="inception_3b")(x)

    # (None, 28, 28, 480)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_3")(x)
    # (None, 14, 14, 480)
    x = InceptionBlk(192, 96, 208, 16, 48, 64, name="inception_4a")(x)
    if is_train:
        aux1 = AuxiliaryBlk(class_num, name="aux_1")(x)

    # (None, 14, 14, 512)
    x = InceptionBlk(160, 112, 224, 24, 64, 64, name="inception_4b")(x)
    # (None, 14, 14, 512)
    x = InceptionBlk(128, 128, 256, 24, 64, 64, name="inception_4c")(x)
    # (None, 14, 14, 512)
    x = InceptionBlk(112, 144, 288, 32, 64, 64, name="inception_4d")(x)
    if is_train:
        aux2 = AuxiliaryBlk(class_num, name="aux_2")(x)

    # (None, 14, 14, 528)
    x = InceptionBlk(256, 160, 320, 32, 128, 128, name="inception_4e")(x)
    # (None, 14, 14, 532)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_4")(x)

    # (None, 7, 7, 832)
    x = InceptionBlk(256, 160, 320, 32, 128, 128, name="inception_5a")(x)
    # (None, 7, 7, 832)
    x = InceptionBlk(384, 192, 384, 48, 128, 128, name="inception_5b")(x)
    # (None, 7, 7, 1024)
    x = layers.AvgPool2D(pool_size=7, strides=1, name="avgpool_1")(x)

    # (None, 1, 1, 1024)
    x = layers.Flatten(name="output_flatten")(x)
    # (None, 1024)
    x = layers.Dropout(rate=0.4, name="output_dropout")(x)
    x = layers.Dense(class_num, name="output_dense")(x)
    # (None, class_num)
    aux3 = layers.Softmax(name="aux_3")(x)

    if is_train:
        model = models.Model(inputs=input_image, outputs=[aux1, aux2, aux3])
    else:
        model = models.Model(inputs=input_image, outputs=aux3)
    return model
