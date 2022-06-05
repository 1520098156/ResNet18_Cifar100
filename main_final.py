import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from net import ResNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(50)

validation_images = train_images[45000:]
train_images = train_images[0:45000]
validation_labels = train_labels[45000:]
train_labels = train_labels[0:45000]

datagen = ImageDataGenerator(
    featurewise_center=False,  # 布尔值。将输入数据的均值设置为 0，逐特征进行。
    samplewise_center=False,  # 布尔值。将每个样本的均值设置为 0。
    featurewise_std_normalization=False,  # 布尔值。将输入除以数据标准差，逐特征进行。
    samplewise_std_normalization=False,  # 布尔值。将每个输入除以其标准差。
    zca_whitening=False,  # 布尔值。是否应用 ZCA 白化。
    # zca_epsilon  ZCA 白化的 epsilon 值，默认为 1e-6。
    # rotation_range=30,  # 整数。随机旋转的度数范围 (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # 布尔值。随机水平翻转。
    vertical_flip=False,  # 布尔值。随机垂直翻转
    fill_mode='nearest'
)

model = ResNet([2, 2, 2, 2])


def spareCE(y_true, y_pred):
    sce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return tf.reduce_mean(sce(y_true, y_pred))


def l2_loss(my_model, weights=1e-4):
    variable_list = []
    for v in my_model.trainable_variables:
        if 'kernel' or 'bias' in v.name:
            variable_list.append(tf.nn.l2_loss(v))
    return tf.add_n(variable_list) * weights


def myLoss(y_true, y_pred):
    sce = spareCE(y_true, y_pred)
    l2 = l2_loss(my_model=model)
    loss = sce + l2
    return loss


if __name__ == '__main__':
    datagen.fit(train_images)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.004),
        loss=myLoss,
        metrics=['sparse_categorical_accuracy'])

    history = model.fit(
        datagen.flow(train_images, train_labels, batch_size=64), steps_per_epoch=len(train_images) / 64,
        epochs=150, verbose=2,
        validation_data=(validation_images, validation_labels)
    )

    model.save('saved_model/my_model')

    fig1, ax_acc = plt.subplots()
    plt.plot(history.history['sparse_categorical_accuracy'], 'r', label='acc')
    plt.plot(history.history['val_sparse_categorical_accuracy'], 'b', label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model - Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    fig2, ax_loss = plt.subplots()
    plt.plot(history.history['loss'], 'r', label='loss')
    plt.plot(history.history['val_loss'], 'b', label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model- Loss')
    plt.legend(loc='upper right')
    plt.show()
