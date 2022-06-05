import cv2 as cv
import numpy as np
import tensorflow as tf

# 精细类别的序号与名称 序号:名称
fineLabelNameDict = {}
# 精细类别对应的粗糙类别 精细序号：粗糙序号-粗糙名称
fineLableToCoraseLabelDict = {}


def myLoss(y_true, y_pred):
    sce = spareCE(y_true, y_pred)
    l2 = l2_loss(my_model=my_model)
    loss = sce + l2
    return loss


my_model = tf.keras.models.load_model('saved_model/my_model', custom_objects={'myLoss': myLoss})
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(50)


def spareCE(y_true, y_pred):
    sce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return tf.reduce_mean(sce(y_true, y_pred))


def l2_loss(my_model, weights=1e-4):
    variable_list = []
    for v in my_model.trainable_variables:
        if 'kernel' or 'bias' in v.name:
            variable_list.append(tf.nn.l2_loss(v))
    return tf.add_n(variable_list) * weights


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def dealData(meta, train):
    for fineLabel, coarseLabel in zip(train[b'fine_labels'], train[b'coarse_labels']):
        if fineLabel not in fineLabelNameDict.keys():
            fineLabelNameDict[fineLabel] = meta[b'fine_label_names'][fineLabel].decode('utf-8')
        if fineLabel not in fineLableToCoraseLabelDict.keys():
            fineLableToCoraseLabelDict[fineLabel] = str(coarseLabel) + "-" + meta[b'coarse_label_names'][
                coarseLabel].decode('utf-8')


if __name__ == '__main__':
    meta = unpickle('cifar-100-python/meta')
    train = unpickle('cifar-100-python/test')
    dealData(meta, train)

    img = np.zeros([3300, 6400, 3], np.uint8)
    font = cv.FONT_HERSHEY_SIMPLEX
    for i in range(100):
        category = fineLabelNameDict[i]
        cv.putText(img, text=category, org=(10, i * 32 + 50), fontFace=font, fontScale=1, color=(255, 255, 255),
                   lineType=cv.LINE_AA)

    position = np.zeros([100], dtype=int)
    for x, y in test_dataset:
        y_pred = my_model(x, training=False)
        y_pred = tf.cast(tf.argmax(y_pred, 1), dtype=tf.int32)
        for image, img_category in zip(x, y_pred):
            print(img_category, '__', position[img_category])
            x0 = int(img_category) * 32 + 18
            y0 = position[img_category] * 32 + 250
            image = image[:, :, ::-1]  # RGB TO BGR
            img[x0:x0 + image.shape[0], y0:y0 + image.shape[1], :] = image * 255.0
            position[img_category] += 1

    # cv.imshow('img', img)
    cv.imwrite('test_visualisation.png', img=img)
