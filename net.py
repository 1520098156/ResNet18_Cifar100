import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import layers


class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.left = keras.Sequential([
            layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=1, padding='same'),
            layers.BatchNormalization()
        ])
        if stride != 1:
            self.downSample = keras.Sequential([
                layers.Conv2D(filters=filter_num, kernel_size=(1, 1), strides=stride),
                layers.BatchNormalization()
            ])
        else:
            self.downSample = lambda x: x

    def call(self, inputs, training=None):
        identity = self.downSample(inputs)
        out = self.left(inputs)
        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        return output


def make_res_block(filter_num, blocks, stride=1):
    res_block = keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride))
    for i in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))
    return res_block


class ResNet(keras.Model):
    def __init__(self, layer_dims, num_class=100):
        super(ResNet, self).__init__()
        self.pre = keras.Sequential([
            layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')
        ])

        self.layer1 = make_res_block(64, layer_dims[0])
        self.layer2 = make_res_block(128, layer_dims[1], stride=2)
        self.layer3 = make_res_block(256, layer_dims[2], stride=2)
        self.layer4 = make_res_block(512, layer_dims[3], stride=2)

        self.average_pooling = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_class)

    def call(self, inputs, training=None):
        x = self.pre(inputs, training=training)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.average_pooling(x)
        x = self.fc(x)

        return x
