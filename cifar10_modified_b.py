# Model B from https://arxiv.org/pdf/1412.6806.pdf
import cPickle
import os
import pickle
import tarfile

import numpy as np
import six
import tensorflow as tf
from six.moves import range

import dataset_class

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './cifar10_data', """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_string('summaries_dir', './cifar10_modified_b', 'Summaries directory')
tf.app.flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')

# prevents multiple graphs warning in tensorboard
if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
tf.gfile.MakeDirs(FLAGS.summaries_dir)

# Creating a decay for the learning rate at step 200, 250 and 300
weight_decay = 0.001
epoch_per_decay1 = 200
epoch_per_decay2 = 250
epoch_per_decay3 = 300
lr_decay_factor = 0.1
init_lr = 0.05

num_batches_per_epoch = 50000 / FLAGS.batch_size
decay_steps_1 = int(num_batches_per_epoch * epoch_per_decay1)
decay_steps_2 = int(num_batches_per_epoch * epoch_per_decay2)
decay_steps_3 = int(num_batches_per_epoch * epoch_per_decay3)

global_step = tf.Variable(0, trainable=False)
decayed_lr_1 = tf.train.exponential_decay(init_lr, global_step, decay_steps_1, lr_decay_factor, staircase=True)
decayed_lr_2 = tf.train.exponential_decay(decayed_lr_1, global_step, decay_steps_2, lr_decay_factor, staircase=True)
decayed_lr = tf.train.exponential_decay(decayed_lr_2, global_step, decay_steps_3, lr_decay_factor, staircase=True)


# Applying global contrast normalize followed by whitening
def global_contrast_normalize(input_x, scale=1., min_divisor=1e-8):
    input_x = input_x - input_x.mean(axis=1)[:, np.newaxis]
    normalizers = np.sqrt((input_x ** 2).sum(axis=1)) / scale
    normalizers[normalizers < min_divisor] = 1.
    input_x /= normalizers[:, np.newaxis]
    return input_x


def compute_zca_transform(images, filter_bias=0.1):
    mean_x = np.mean(images, 0)
    cov_x = np.cov(images.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_x + filter_bias * np.eye(cov_x.shape[0], cov_x.shape[1]))
    assert not np.isnan(eigenvalues).any()
    assert not np.isnan(eigenvectors).any()
    assert eigenvalues.min() > 0
    eigenvalues = eigenvalues ** -0.5
    whiten = np.dot(eigenvectors, np.dot(np.diag(eigenvalues), eigenvectors.T))
    return mean_x, whiten


def zca_whiten(train, test, cache=None):
    if cache and os.path.isfile(cache):
        with open(cache, 'rb') as f:
            (mean_x, whiten) = pickle.load(f)
    else:
        mean_x, whiten = compute_zca_transform(train)

        with open(cache, 'wb') as f:
            pickle.dump((mean_x, whiten), f, 2)

    train_whiten = np.dot(train - mean_x, whiten)
    test_whiten = np.dot(test - mean_x, whiten)
    return train_whiten, test_whiten


def _variable_with_weight_decay(shape, wd=weight_decay):
    initial = tf.random_normal(shape, stddev=0.05)
    var = tf.Variable(initial)
    if wd is not None:
        wd = tf.multiply(tf.nn.l2_loss(var), wd, name="weight_loss")
        tf.add_to_collection("losses", wd)
    return var


def unpickle(f):
    fo = open(f, 'rb')
    dictionary = cPickle.load(fo)
    fo.close()
    return dictionary


def load_cifar_10():
    # LOAD TRAINING DATA
    tar_file = tarfile.open("cifar-10-python.tar.gz", 'r:gz')
    train_batches = []
    for index in range(1, 6):
        f = tar_file.extractfile('cifar-10-batches-py/data_batch_%d' % index)
        try:
            if six.PY3:
                array = cPickle.load(f, encoding='latin1')
            else:
                array = cPickle.load(f)
            train_batches.append(array)
        finally:
            f.close()
    tr_x = np.concatenate([x_batch['data'] for x_batch in train_batches], axis=0)
    tr_y = np.concatenate([np.array(y_batch['labels'], dtype=np.int8) for y_batch in train_batches], axis=0)

    # LOAD TEST DATA
    f = tar_file.extractfile('cifar-10-batches-py/test_batch')
    try:
        if six.PY3:
            test = cPickle.load(f, encoding='latin1')
        else:
            test = cPickle.load(f)
    finally:
        f.close()
    te_x = test['data']
    te_y = np.array(test['labels'], dtype=np.int8)

    # one-hot encoding of labels
    tr_labels = np.zeros((50000, 10), dtype=np.float32)
    te_labels = np.zeros((10000, 10), dtype=np.float32)

    for i in range(50000):
        a = tr_y[i]
        tr_labels[i, a] = 1.

    for j in range(10000):
        b = te_y[j]
        te_labels[j, b] = 1.

    return tr_x, tr_labels, te_x, te_labels


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")


def conv2d_s1_same(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def conv2d_s2_same(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")


def average_pool(x):
    return tf.nn.avg_pool(x, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="VALID")


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y-input")

with tf.name_scope('dropout'):
    keep_prob_input = tf.placeholder(tf.float32)
    keep_prob_pool = tf.placeholder(tf.float32)

# 3x3 conv 96 ReLU
with tf.name_scope('conv1'):
    x_dropped = tf.nn.dropout(x, keep_prob_input)
    W_conv1 = _variable_with_weight_decay([5, 5, 3, 96])
    b_conv1 = _variable_with_weight_decay([96])
    conv1 = tf.nn.relu(tf.nn.bias_add(conv2d_s1_same(x_dropped, W_conv1), b_conv1))

# 3x3 conv 96 ReLU
with tf.name_scope('conv2'):
    W_conv2 = _variable_with_weight_decay([1, 1, 96, 96])
    b_conv2 = _variable_with_weight_decay([96])
    conv2 = tf.nn.relu(tf.nn.bias_add(conv2d_s1_same(conv1, W_conv2), b_conv2))

# 3x3 conv 96 ReLU with stride = 2
with tf.name_scope('conv3'):
    W_conv3 = _variable_with_weight_decay([3, 3, 96, 96])
    b_conv3 = _variable_with_weight_decay([96])
    conv3 = tf.nn.relu(tf.nn.bias_add(conv2d_s2_same(conv2, W_conv3), b_conv3))
    conv3_dropped = tf.nn.dropout(conv3, keep_prob=keep_prob_pool)

# 3x3 conv 192 ReLU
with tf.name_scope('conv4'):
    W_conv4 = _variable_with_weight_decay([5, 5, 96, 192])
    b_conv4 = _variable_with_weight_decay([192])
    conv4 = tf.nn.relu(tf.nn.bias_add(conv2d_s1_same(conv3_dropped, W_conv4), b_conv4))

# 3x3 conv 192 ReLU
with tf.name_scope('conv5'):
    W_conv5 = _variable_with_weight_decay([1, 1, 192, 192])
    b_conv5 = _variable_with_weight_decay([192])
    conv5 = tf.nn.relu(tf.nn.bias_add(conv2d_s1_same(conv4, W_conv5), b_conv5))

# 3x3 conv 192 ReLU with stride = 2
with tf.name_scope('conv6'):
    W_conv6 = _variable_with_weight_decay([3, 3, 192, 192])
    b_conv6 = _variable_with_weight_decay([192])
    conv6 = tf.nn.relu(tf.nn.bias_add(conv2d_s2_same(conv5, W_conv6), b_conv6))
    conv6_dropped = tf.nn.dropout(conv6, keep_prob=keep_prob_pool)

# 3x3 conv 192 ReLU
with tf.name_scope('conv7'):
    W_conv7 = _variable_with_weight_decay([3, 3, 192, 192])
    b_conv7 = _variable_with_weight_decay([192])
    conv7 = tf.nn.relu(tf.nn.bias_add(conv2d_s1_same(conv6_dropped, W_conv7), b_conv7))

# 1x1 conv 192 ReLU
with tf.name_scope('conv8'):
    W_conv8 = _variable_with_weight_decay([1, 1, 192, 192])
    b_conv8 = _variable_with_weight_decay([192])
    conv8 = tf.nn.relu(tf.nn.bias_add(conv2d(conv7, W_conv8), b_conv8))

# 1x1 conv 10 ReLU
with tf.name_scope('conv9'):
    W_conv9 = _variable_with_weight_decay([1, 1, 192, 10])
    b_conv9 = _variable_with_weight_decay([10])
    conv9 = tf.nn.relu(tf.nn.bias_add(conv2d(conv8, W_conv9), b_conv9))

with tf.name_scope('output'):
    logits = average_pool(conv9)
    pred = tf.nn.softmax(tf.reshape(logits, (-1, 10)))

with tf.name_scope("loss"):
    cross_entropy_mean = -tf.reduce_mean(tf.reduce_sum(y_ * tf.log(pred + np.exp(-50.0)), 1))
    tf.add_to_collection("total_loss", cross_entropy_mean)
    loss = tf.add_n(tf.get_collection("total_loss"), "total_loss")
    ce_summ = tf.summary.scalar("loss", loss)

# using stochastic gradient descent with fixed momentum of 0.9
with tf.name_scope("train"):
    train_step = tf.train.MomentumOptimizer(decayed_lr, 0.9).minimize(loss, global_step)

with tf.name_scope("learning_rate") as scope:
    tf.summary.scalar('learning_rate', decayed_lr)

with tf.name_scope("accuracy"):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), 0)
    accuracy_summary = tf.summary.scalar("accuracy", accuracy)

saver = tf.train.Saver()
sess = tf.Session()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

X_train, y_train, X_test, y_test = load_cifar_10()
X_train = global_contrast_normalize(X_train, scale=55.0)
X_test = global_contrast_normalize(X_test, scale=55.0)
X_train, X_test = zca_whiten(X_train, X_test, cache='./cifar10-zca-cache.pkl')

# Reformatting data as images
X_train = X_train.reshape((X_train.shape[0], 3, 32, 32)).transpose((0, 2, 3, 1))
X_test = X_test.reshape((X_test.shape[0], 3, 32, 32)).transpose((0, 2, 3, 1))

cifar10 = dataset_class.read_data_sets(X_train, y_train, X_test, y_test, 0)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(FLAGS.max_steps):
        batch = cifar10.train.next_batch(FLAGS.batch_size)
        global_step = i
        if i % 100 == 0:
            test_batch = cifar10.test.next_batch(FLAGS.batch_size)
            test_summary, test_acc = sess.run([merged, accuracy], feed_dict={x: test_batch[0], y_: test_batch[1],
                                                                             keep_prob_input: 1, keep_prob_pool: 1})
            test_writer.add_summary(test_summary, i)
            acc_batch, loss_batch = sess.run([accuracy, loss], feed_dict={x: batch[0], y_: batch[1],
                                                                          keep_prob_input: 1.0, keep_prob_pool: 1.0})
            print "Step: %s, Train Acc: %s, Loss: %s; Test Acc: %s" % (i, acc_batch, loss_batch, test_acc)
        else:
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1],
                                                                   keep_prob_input: 0.8, keep_prob_pool: 0.5})
            train_writer.add_summary(summary, i)


    final_test_accuracy = 0.
    for i in xrange(0, 10):
        final_test_accuracy += 0.1 * sess.run(accuracy, feed_dict={x: cifar10.test.images[i * 1000:(i + 1) * 1000],
                                                                   y_: cifar10.test.labels[i * 1000:(i + 1) * 1000],
                                                                   keep_prob_input: 1.0, keep_prob_pool: 1.0})

    print "Final accuracy on Test set: ", final_test_accuracy

save_path = saver.save(sess, "./cifar10_modified_model_b.ckpt")
print "Model saved in file: ", save_path
tf.Session.close(sess)
