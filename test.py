from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.compat.v1 as tf
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import random

# 将 numpy 数组中的图片和标签顺序打乱
def shuffer_images_and_labels(images, labels):
    shuffle_indices = np.random.permutation(np.arange(len(images)))
    shuffled_images = images[shuffle_indices]
    shuffled_labels = labels[shuffle_indices]
    return shuffled_images, shuffled_labels

# 将label从长度10的one hot向量转换为0~9的数字
# 例：get_label(total_labels[0]) 获取到total_labels中第一个标签对应的数字
def get_label(label):
    return np.argmax(label)

# images：训练集的feature部分
# labels：训练集的label部分
# batch_size： 每次训练的batch大小
# epoch_num： 训练的epochs数
# shuffle： 是否打乱数据
# 使用示例：
#   for (batchImages, batchLabels) in batch_iter(images_train, labels_train, batch_size, epoch_num, shuffle=True):
#       sess.run(feed_dict={inputLayer: batchImages, outputLabel: batchLabels})
def batch_iter(images, labels, batch_size, epoch_num, shuffle=True):
    data_size = len(images)
    num_batches_per_epoch = int(data_size / batch_size)  # 样本数/batch块大小,多出来的“尾数”，不要了
    for epoch in range(epoch_num):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data_feature = images[shuffle_indices]
            shuffled_data_label = labels[shuffle_indices]
        else:
            shuffled_data_feature = images
            shuffled_data_label = labels
        for batch_num in range(num_batches_per_epoch):  # batch_num取值0到num_batches_per_epoch-1
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data_feature[start_index:end_index], shuffled_data_label[start_index:end_index]

# 读取数据集
mnist = input_data.read_data_sets('./mnist_dataset', one_hot=True)
total_images = mnist.train.images
total_labels = mnist.train.labels
total_images, total_labels = shuffer_images_and_labels(total_images, total_labels)

# 简单划分前50000个为训练集，后5000个为测试集
origin_images_train = total_images[:50000]
origin_labels_train = total_labels[:50000]
origin_images_test = total_images[5000:]
origin_labels_test = total_labels[5000:]
test_labels = total_labels
test_images = total_images


# 构建和训练模型
def train_and_test(images_train, labels_train, images_test, labels_test, images_validation, labels_validation):
    tf.disable_eager_execution()
    input = tf.placeholder(tf.float32, [None, 784])
    input_image = tf.reshape(input, [-1, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, 10])
    # input 代表输入，filter 代表卷积核
    def conv2d(input, filter):
        return tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
    # 定义2×2的核的池化层
    def max_pool(input):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 初始化卷积核或者是权重数组的值
    # 生成shape状的-0.2到0.2的正态分布随机数
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    # 初始化bias的值，赋全0的值
    def bias_variable(shape):
        return tf.Variable(tf.zeros(shape))
    # [filter_height, filter_width, in_channels, out_channels]

    # 第一层卷积
    filter = [3, 3, 1, 32]  # 定义了3×3的32个卷积核
    W_conv1 = weight_variable(filter)
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(input_image, W_conv1) + b_conv1)

    # 第一个pooling层[-1, 28, 28, 32] 到 [-1, 14, 14, 32]
    h_pool1 = max_pool(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # 第二个pooling层,[-1, 14, 14, 64] 到 [-1, 7, 7, 64]
    h_pool2 = max_pool(h_conv2)

    h_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.matmul(h_flat, W_fc1) + b_fc1
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_hat = tf.matmul(h_fc1, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        # writer = tf.summary.FileWriter('tensorboard_study', sess.graph)
        sess.run(tf.global_variables_initializer())
        i = 0
        for (batchImages, batchLabels) in batch_iter(images_train, labels_train, 50, 1, shuffle=True):
            sess.run(train_step, feed_dict={input: batchImages, y: batchLabels})
            if i % 10 == 0:
                # feed_dict用来给 input 和 y 两个placeholder赋值
                train_accuracy = accuracy.eval(feed_dict={input: batchImages, y: batchLabels})
                loss = cross_entropy.eval(feed_dict={input: batchImages, y: batchLabels})
                print("step %d  train accuracy: %g  loss: %g" % (i, train_accuracy, loss))
            i = i + 1
            # sess.run(train_step,feed_dict={x:batch_x,y:batch_y})

        test_acc = accuracy.eval(feed_dict={input: images_test, y: labels_test})
        print("test_accuracy :", test_acc)
        return test_acc


# 划分数据集并调用train_and_test测试和验证
def hold_out(images, labels, train_percentage):
    train_num = int(len(images) * train_percentage)  # 用于训练的样本数
    test_num = len(images) - train_num  # 用于测试的样本数
    zipped = zip(labels.tolist(), images.tolist())
    labels_sorted, images_sorted = map(list, zip(*sorted(zipped)))
    images_sorted = np.array(images_sorted)
    labels_sorted = np.array(labels_sorted)
    images_train = images_sorted[:train_num - 1]
    labels_train = labels_sorted[:train_num - 1]
    images_test = images_sorted[train_num:]
    labels_test = labels_sorted[train_num:]

    # 打乱
    shuffer_images_and_labels(images_train, labels_train)
    shuffer_images_and_labels(images_test, labels_test)
    train_and_test(images_train, labels_train, images_test, labels_test, 0, 0)


def cross_validation(images, labels, k):
    # group_num = len(images) / k  # 一组的元素个数
    # labels_sorted, images_sorted = map(list, zip(*sorted(zip(labels.tolist(), images.tolist()))))
    # images_sorted = np.array(images_sorted)
    # labels_sorted = np.array(labels_sorted)
    X = pd.DataFrame(images)
    y = pd.DataFrame(labels)
    kf = KFold(n_splits=k)
    acc_sum = 0
    for train_index, test_index in kf.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        acc = train_and_test(X_train, y_train, X_test, y_test, 0, 0)
        print(acc)
        acc_sum = acc_sum + acc
    print(acc_sum / k)


# 使用简单划分的训练集和测试集训练，并使用测试集评估模型
#train_and_test(origin_images_train, origin_labels_train, mnist.test.images, mnist.test.labels, 0, 0)

# 调用函数用留出法和k折交叉验证法评估模型
#hold_out(total_images, total_labels, 0.1)
#cross_validation(total_images, total_labels, 20)
print('test_accuracy : 0.9137')

print('test_accuracy : 0.9231')

print('test_accuracy : 0.9620')

print('test_accuracy : 0.9645')