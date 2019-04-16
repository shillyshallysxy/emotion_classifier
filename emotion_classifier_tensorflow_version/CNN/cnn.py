import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers


class CNN_Model():
    def __init__(self, num_tags_=7, lr_=0.001, channel_=1, hidden_dim_=1024, full_shape_=2304, optimizer_='Adam'):
        self.num_tags = num_tags_  # 分为几类（七种表情所以分为7类）
        self.lr = lr_  # 学习率
        self.full_shape = full_shape_  # get_shape使用不了，手动计算，权宜之计
        self.channel = channel_  # 输入图像的通道数黑白为1，rgb图为3
        self.hidden_dim = hidden_dim_ # 第一个全连接层的hidden_dim
        self.conv_feature = [32, 32, 32, 64]
        self.conv_size = [1, 5, 3, 5]
        self.maxpool_size = [0, 3, 3, 3]
        self.maxpool_stride = [0, 2, 2, 2]
        # self.initializer = tf.truncated_normal_initializer(stddev=0.05)
        self.initializer = initializers.xavier_initializer()
        self.dropout = tf.placeholder(dtype=tf.float32, name='dropout')
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.channel], name='x_input')
        self.y_target = tf.placeholder(dtype=tf.int32, shape=[None], name='y_target')
        self.batch_size = tf.shape(self.x_input)[0]
        self.logits = self.project_layer(self.cnn_layer())
#         self.logits = self.res_net_layer()
        with tf.variable_scope("loss"):
            self.loss = self.loss_layer(self.logits)
            self.train_step = self.optimizer(self.loss, optimizer_)

    # 卷积层部分
    def cnn_layer(self):
        with tf.variable_scope("conv1"):
            conv1_weight = tf.get_variable('conv1_weight', [self.conv_size[0], self.conv_size[0],
                                                            self.channel, self.conv_feature[0]],
                                           dtype=tf.float32, initializer=self.initializer)
            conv1_bias = tf.get_variable('conv1_bias', [self.conv_feature[0]], dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.conv2d(self.x_input, conv1_weight, [1, 1, 1, 1], padding='SAME')
            conv1_add_bias = tf.nn.bias_add(conv1, conv1_bias)
            conv1_relu = tf.nn.relu(conv1_add_bias)
            norm1 = tf.nn.lrn(conv1_relu, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm1')
        with tf.variable_scope("conv2"):
            conv2_weight = tf.get_variable('conv2_weight', [self.conv_size[1], self.conv_size[1],
                                                            self.conv_feature[0], self.conv_feature[1]],
                                           dtype=tf.float32, initializer=self.initializer)
            conv2_bias = tf.get_variable('conv2_bias', [self.conv_feature[1]], dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.conv2d(norm1, conv2_weight, [1, 1, 1, 1], padding='SAME')
            conv2_add_bias = tf.nn.bias_add(conv2, conv2_bias)
            conv2_relu = tf.nn.relu(conv2_add_bias)
            pool2 = tf.nn.max_pool(conv2_relu, ksize=[1, self.maxpool_size[1], self.maxpool_size[1], 1],
                                   strides=[1, self.maxpool_stride[1], self.maxpool_stride[1], 1],
                                   padding='SAME', name='pool_layer2')
            norm2 = tf.nn.lrn(pool2, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm2')
        with tf.variable_scope("conv3"):
            conv3_weight = tf.get_variable('conv3_weight', [self.conv_size[2], self.conv_size[2],
                                                            self.conv_feature[1], self.conv_feature[2]],
                                           dtype=tf.float32, initializer=self.initializer)
            conv3_bias = tf.get_variable('conv3_bias', [self.conv_feature[2]], dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0))
            conv3 = tf.nn.conv2d(norm2, conv3_weight, [1, 1, 1, 1], padding='SAME')
            conv3_add_bias = tf.nn.bias_add(conv3, conv3_bias)
            conv3_relu = tf.nn.relu(conv3_add_bias)
            pool3 = tf.nn.max_pool(conv3_relu, ksize=[1, self.maxpool_size[2], self.maxpool_size[2], 1],
                                   strides=[1, self.maxpool_stride[2], self.maxpool_stride[2], 1],
                                   padding='SAME', name='pool_layer3')
            norm3 = tf.nn.lrn(pool3, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm3')
        with tf.variable_scope("conv4"):
            conv4_weight = tf.get_variable('conv4_weight', [self.conv_size[3], self.conv_size[3],
                                                            self.conv_feature[2], self.conv_feature[3]],
                                           dtype=tf.float32, initializer=self.initializer)
            conv4_bias = tf.get_variable('conv4_bias', [self.conv_feature[3]], dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0))
            conv4 = tf.nn.conv2d(norm3, conv4_weight, [1, 1, 1, 1], padding='SAME')
            conv4_add_bias = tf.nn.bias_add(conv4, conv4_bias)
            conv4_relu = tf.nn.relu(conv4_add_bias)
            pool4 = tf.nn.max_pool(conv4_relu, ksize=[1, self.maxpool_size[3], self.maxpool_size[3], 1],
                                   strides=[1, self.maxpool_stride[3], self.maxpool_stride[3], 1],
                                   padding='SAME', name='pool_layer4')
            norm4 = tf.nn.lrn(pool4, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm4')
        return norm4

    # 略微简单的卷积层（和上面一个二选一）
    def cnn_layer_single(self):
        with tf.variable_scope("conv1"):
            conv1_weight = tf.get_variable('conv1_weight', [self.conv_size[0], self.conv_size[0],
                                                            self.channel, self.conv_feature[0]],
                                           dtype=tf.float32, initializer=self.initializer)
            conv1_bias = tf.get_variable('conv1_bias', [self.conv_feature[0]], dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.conv2d(self.x_input, conv1_weight, [1, 1, 1, 1], padding='SAME')
            conv1_add_bias = tf.nn.bias_add(conv1, conv1_bias)
            conv1_relu = tf.nn.relu(conv1_add_bias)
        with tf.variable_scope("conv2"):
            conv2_weight = tf.get_variable('conv2_weight', [self.conv_size[1], self.conv_size[1],
                                                            self.conv_feature[0], self.conv_feature[1]],
                                           dtype=tf.float32, initializer=self.initializer)
            conv2_bias = tf.get_variable('conv2_bias', [self.conv_feature[1]], dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.conv2d(conv1_relu, conv2_weight, [1, 1, 1, 1], padding='SAME')
            conv2_add_bias = tf.nn.bias_add(conv2, conv2_bias)
            conv2_relu = tf.nn.relu(conv2_add_bias)
            pool2 = tf.nn.max_pool(conv2_relu, ksize=[1, self.maxpool_size[1], self.maxpool_size[1], 1],
                                   strides=[1, self.maxpool_stride[1], self.maxpool_stride[1], 1],
                                   padding='SAME', name='pool_layer2')
        with tf.variable_scope("conv3"):
            conv3_weight = tf.get_variable('conv3_weight', [self.conv_size[2], self.conv_size[2],
                                                            self.conv_feature[1], self.conv_feature[2]],
                                           dtype=tf.float32, initializer=self.initializer)
            conv3_bias = tf.get_variable('conv3_bias', [self.conv_feature[2]], dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0))
            conv3 = tf.nn.conv2d(pool2, conv3_weight, [1, 1, 1, 1], padding='SAME')
            conv3_add_bias = tf.nn.bias_add(conv3, conv3_bias)
            conv3_relu = tf.nn.relu(conv3_add_bias)
            pool3 = tf.nn.max_pool(conv3_relu, ksize=[1, self.maxpool_size[2], self.maxpool_size[2], 1],
                                   strides=[1, self.maxpool_stride[2], self.maxpool_stride[2], 1],
                                   padding='SAME', name='pool_layer3')
        with tf.variable_scope("conv4"):
            conv4_weight = tf.get_variable('conv4_weight', [self.conv_size[3], self.conv_size[3],
                                                            self.conv_feature[2], self.conv_feature[3]],
                                           dtype=tf.float32, initializer=self.initializer)
            conv4_bias = tf.get_variable('conv4_bias', [self.conv_feature[3]], dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0))
            conv4 = tf.nn.conv2d(pool3, conv4_weight, [1, 1, 1, 1], padding='SAME')
            conv4_add_bias = tf.nn.bias_add(conv4, conv4_bias)
            conv4_relu = tf.nn.relu(conv4_add_bias)
            pool4 = tf.nn.max_pool(conv4_relu, ksize=[1, self.maxpool_size[3], self.maxpool_size[3], 1],
                                   strides=[1, self.maxpool_stride[3], self.maxpool_stride[3], 1],
                                   padding='SAME', name='pool_layer4')
        return pool4

    # 全连接层
    def project_layer(self, x_in_):
        with tf.variable_scope("project"):
            with tf.variable_scope("hidden"):
                x_in_ = tf.reshape(x_in_, [self.batch_size, -1])
                w_tanh1 = tf.get_variable("w_tanh1", [self.full_shape, self.hidden_dim*2], initializer=self.initializer,
                                          regularizer=tf.contrib.layers.l2_regularizer(0.001))
                b_tanh1 = tf.get_variable("b_tanh1", [self.hidden_dim*2], initializer=tf.zeros_initializer())
                w_tanh2 = tf.get_variable("w_tanh2", [self.hidden_dim*2, self.hidden_dim], initializer=self.initializer,
                                          regularizer=tf.contrib.layers.l2_regularizer(0.001))
                b_tanh2 = tf.get_variable("b_tanh2", [self.hidden_dim], initializer=tf.zeros_initializer())
                output1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(x_in_, w_tanh1),
                                                          b_tanh1)), keep_prob=self.dropout)
                output2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(output1, w_tanh2),
                                                          b_tanh2)), keep_prob=self.dropout)
            with tf.variable_scope("output"):
                w_out = tf.get_variable("w_out", [self.hidden_dim, self.num_tags], initializer=self.initializer,
                                        regularizer=tf.contrib.layers.l2_regularizer(0.001))
                b_out = tf.get_variable("b_out", [self.num_tags], initializer=tf.zeros_initializer())
                pred_ = tf.add(tf.matmul(output2, w_out), b_out, name='logits')
        return pred_
    
    # resnet部分，如果使用则不需要使用上面的cnn和project部分
    def res_net_layer(self):
        print('Using Res Net ')
        with tf.variable_scope("resnet"):
            conv1_weight = tf.get_variable('conv1_weight', [5, 5, self.channel, 32],
                                           dtype=tf.float32, initializer=self.initializer)
            conv1 = tf.nn.conv2d(self.x_input, conv1_weight, [1, 1, 1, 1], padding='SAME')
            conv1 = tf.layers.batch_normalization(conv1, training=self.is_training)
            conv1 = tf.nn.relu(conv1)
            conv1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_layer')
            resnet = resnet_utils.block_net(conv1, 32, [32, 32], self.is_training, 1, 'resnet1')
            resnet = resnet_utils.block_net(resnet, 32, [32, 64], self.is_training, 2, 'resnet2')
            resnet = resnet_utils.block_net(resnet, 64, [64, 64], self.is_training, 1, 'resnet3')
            resnet = resnet_utils.block_net(resnet, 64, [64, 128], self.is_training, 2, 'resnet4')
            resnet = resnet_utils.block_net(resnet, 128, [128, 128], self.is_training, 1, 'resnet5')
            resnet = resnet_utils.block_net(resnet, 128, [128, 256], self.is_training, 2, 'resnet6')
            resnet = resnet_utils.block_net(resnet, 256, [256, 256], self.is_training, 1, 'resnet7')
            # resnet = resnet_utils.bottleneck(resnet, 256, [64, 64, 512], self.is_training, 2, 'resnet8')
            pool = tf.nn.avg_pool(resnet, [1, 3, 3, 1], [1, 1, 1, 1], padding='VALID')

            # convf_weight = tf.get_variable('convf_weight', [3, 3, 256, 256],
            #                                dtype=tf.float32, initializer=self.initializer)
            # convf = tf.nn.conv2d(resnet, convf_weight, [1, 1, 1, 1], padding='VALID')
            # convf = tf.layers.batch_normalization(convf, training=self.is_training)

            project_in = pool
            project_in = tf.reshape(project_in, [self.batch_size, -1])
        with tf.variable_scope("project"):
            with tf.variable_scope("output"):
                w_tanh1 = tf.get_variable("w_tanh1", [256, self.num_tags], initializer=self.initializer,
                                          regularizer=tf.contrib.layers.l2_regularizer(0.001))
                b_tanh1 = tf.get_variable("b_tanh1", [self.num_tags], initializer=tf.zeros_initializer())
                output = tf.add(tf.matmul(project_in, w_tanh1), b_tanh1, name='logits')
        return output    
    
    def loss_layer(self, project_logits):
        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=project_logits, labels=self.y_target), name='softmax_loss')
        return loss

    def optimizer(self, loss_, method=''):
        if method == 'Momentum':
            step = tf.Variable(0, trainable=False)
            model_learning_rate = tf.train.exponential_decay(0.01, step,
                                                             100, 0.99, staircase=True)
            my_optimizer = tf.train.MomentumOptimizer(model_learning_rate, momentum=0.9)
            train_step_ = my_optimizer.minimize(loss_, global_step=step, name='train_step')
            print('Using ', method)
        elif method == 'SGD':
            step = tf.Variable(0, trainable=False)
            model_learning_rate = tf.train.exponential_decay(0.1, step,
                                                             200., 0.96, staircase=True)
            my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
            train_step_ = my_optimizer.minimize(loss_, name='train_step')
            print('Using ', method)
        elif method == 'Adam':
            train_step_ = tf.train.AdamOptimizer(self.lr).minimize(loss_, name='train_step')
            print('Using ', method)
        else:
            train_step_ = tf.train.MomentumOptimizer(0.005, momentum=0.9).minimize(loss_, name='train_step')
            print('Using Default')
        return train_step_
