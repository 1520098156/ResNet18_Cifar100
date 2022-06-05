import numpy as np
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)


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
    l2 = l2_loss(my_model=my_model)
    loss = sce + l2
    return loss


if __name__ == '__main__':
    my_model = tf.keras.models.load_model('saved_model/my_model', custom_objects={'myLoss': myLoss})

    correct_num = 0
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(50)
    for x, y in test_dataset:
        y_pred = my_model(x, training=False)
        y_pred = tf.cast(tf.argmax(y_pred, 1), dtype=tf.int32)
        y_true = tf.cast(tf.squeeze(y, -1), dtype=tf.int32)
        equality = tf.equal(y_pred, y_true)
        equality = tf.cast(equality, dtype=tf.float32)
        correct_num += tf.reduce_sum(equality)
        print(float(correct_num))
    print('acc=', float(correct_num) / 10000.0)
