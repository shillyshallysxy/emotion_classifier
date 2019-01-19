import tensorflow as tf


def bottleneck(input_, feature_input, features, is_training, stride, name):
    f1, f2, f3 = features
    shortcut_input = input_
    with tf.variable_scope(name):
        conv1_weight = tf.get_variable(name='conv1', shape=[1, 1, feature_input, f1],
                                       initializer=tf.truncated_normal_initializer(stddev=0.01))
        input_ = tf.nn.conv2d(input_, conv1_weight, [1, 1, 1, 1], padding='SAME')
        input_ = tf.layers.batch_normalization(input_, training=is_training)
        input_ = tf.nn.relu(input_)

        conv2_weight = tf.get_variable(name='conv2', shape=[3, 3, f1, f2],
                                       initializer=tf.truncated_normal_initializer(stddev=0.01))
        input_ = tf.nn.conv2d(input_, conv2_weight, [1, stride, stride, 1], padding='SAME')
        input_ = tf.layers.batch_normalization(input_, training=is_training)
        input_ = tf.nn.relu(input_)

        conv3_weight = tf.get_variable(name='conv3', shape=[1, 1, f2, f3],
                                       initializer=tf.truncated_normal_initializer(stddev=0.01))
        input_ = tf.nn.conv2d(input_, conv3_weight, [1, 1, 1, 1], padding='SAME')
        input_ = tf.layers.batch_normalization(input_, training=is_training)
        if not (feature_input == f3):
            convs_weight = tf.get_variable(name='convs', shape=[1, 1, feature_input, f3],
                                           initializer=tf.truncated_normal_initializer(stddev=0.01))
            shortcut_input = tf.nn.conv2d(shortcut_input, convs_weight, [1, stride, stride, 1], padding='SAME')
        shortcut_input = tf.layers.batch_normalization(shortcut_input, training=is_training)
        input_ = tf.nn.relu(tf.add(shortcut_input, input_))
    return input_


def block_net(input_, feature_input, features, is_training, stride, name):
    f1, f2 = features
    shortcut_input = input_
    with tf.variable_scope(name):
        conv1_weight = tf.get_variable(name='conv1', shape=[3, 3, feature_input, f1],
                                       initializer=tf.truncated_normal_initializer(stddev=0.01))
        input_ = tf.nn.conv2d(input_, conv1_weight, [1, stride, stride, 1], padding='SAME')
        input_ = tf.layers.batch_normalization(input_, training=is_training)
        input_ = tf.nn.relu(input_)

        conv2_weight = tf.get_variable(name='conv2', shape=[3, 3, f1, f2],
                                       initializer=tf.truncated_normal_initializer(stddev=0.01))
        input_ = tf.nn.conv2d(input_, conv2_weight, [1, 1, 1, 1], padding='SAME')
        input_ = tf.layers.batch_normalization(input_, training=is_training)

        if not (feature_input == f2):
            convs_weight = tf.get_variable(name='convs', shape=[1, 1, feature_input, f2],
                                           initializer=tf.truncated_normal_initializer(stddev=0.01))
            shortcut_input = tf.nn.conv2d(shortcut_input, convs_weight, [1, stride, stride, 1], padding='SAME')
        input_ = tf.nn.relu(tf.add(shortcut_input, input_))
    return input_
