import tensorflow as tf

model_list = ["io-units_x1+cnn_pooling2",
              "io-units_x1+cnn_pooling",
              "io-units_x2",
              "io-layer",
              "io-units_x1",
              "io-units_x1+layer",
              "io-units_x1+units_x1",
              "io-units_x1+units_x2",
              "baseline",
              "original",
             ]
parameters = [189106, 705954, 1391730, 2780306, 2788434, 2788882, 2806034,
              5540946, 5557458, 22170834]

def Conv2D(images, scope_header):
    input_dim = images.get_shape()[-1].value
    with tf.variable_scope(scope_header) as scope:
        weights = tf.get_variable('weights',
                                    shape = [3,3,input_dim,16],
                                    dtype = tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                    shape=[16],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name= scope.name)
        return conv

def MaxPooling2D(conv, scope_header, strides=[2,2]):
    with tf.variable_scope(scope_header) as scope:
        pool = tf.nn.max_pool(conv, ksize=[1,3,3,1],strides=[1,strides[0],strides[1],1],
                                padding='SAME', name=scope.name)
        return pool

def LRN2D(pool, scope_header):
    with tf.variable_scope(scope_header) as scope:
        norm = tf.nn.lrn(pool, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                            beta=0.75, name=scope.name)
        return norm

def Dense(local_in, out_dim, scope_header):
    with tf.variable_scope(scope_header) as scope:
        dim = local_in.get_shape()[1].value
        weights = tf.get_variable('weights',
                                    shape=[dim, out_dim],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                    shape=[out_dim],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
        local_out = tf.nn.relu(tf.matmul(local_in, weights) + biases, name=scope.name)
        return local_out

def Flatten(local_in):
    return tf.reshape(local_in, shape=[local_in.get_shape()[0].value, -1])

def Softmax(local_in, n_classes, scope_header):
    with tf.variable_scope(scope_header) as scope:
        weights = tf.get_variable('softmax_linear',
                                    shape=[local_in.get_shape()[1].value, n_classes],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                    shape=[n_classes],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local_in, weights), biases, name=scope.name)
        return softmax_linear

def get_model(images, batch_size, n_classes, setting=8):
    output = None

    conv1 = Conv2D(images, "conv1")
    pool1 = MaxPooling2D(conv1, "pool1")
    norm1 = LRN2D(pool1, "norm1")

    conv2 = Conv2D(norm1, "conv2")
    if setting == 9:
        pool2 = MaxPooling2D(conv2, "pool2", [1,1])
    else: pool2 = MaxPooling2D(conv2, "pool2")
    norm2 = LRN2D(pool2, "norm2")

    if setting == 8 or setting == 9:
        # [baseline model]
        dense0 = Flatten(norm2)
        dense1 = Dense(dense0, 128, "dense1")
        output = Dense(dense1, 128, "dense2")
    elif setting == 7:
        # [io-units_x1+units_x2 model]
        dense0 = Flatten(norm2)
        dense1 = Dense(dense0, 64, "dense1")
        output = Dense(dense1, 512, "dense2")
    elif setting == 6:
        # [io-units_x1+units_x1 model]
        dense0 = Flatten(norm2)
        dense1 = Dense(dense0, 64, "dense1")
        output = Dense(dense1, 256, "dense2")
    elif setting == 5:
        # [io-units_x1+layer model]
        dense0 = Flatten(norm2)
        dense1 = Dense(dense0, 64, "dense1")
        dense2 = Dense(dense1, 128, "dense2")
        output = Dense(dense2, 64, "dense3")
    elif setting == 4:
        # [io-units_x1 model]
        dense0 = Flatten(norm2)
        dense1 = Dense(dense0, 64, "dense1")
        output = Dense(dense1, 128, "dense2")
    elif setting == 3:
        # [io-layer model]
        dense0 = Flatten(norm2)
        output = Dense(dense0, 128, "dense2")
    elif setting == 2:
        # [io-units_x2]
        dense0 = Flatten(norm2)
        dense1 = Dense(dense0, 32, "dense1")
        output = Dense(dense1, 128, "dense2")
    elif setting == 1:
        # [io-units_x1+cnn_pooling]
        conv3 = Conv2D(norm2, "conv3")
        pool3 = MaxPooling2D(conv3, "pool3")
        norm3 = LRN2D(pool3, "norm3")

        dense0 = Flatten(norm3)
        dense1 = Dense(dense0, 64, "dense1")
        output = Dense(dense1, 128, "dense2")
    else:
        # [io-units_x1+cnn_pooling2 model]
        conv3 = Conv2D(norm2, "conv3")
        pool3 = MaxPooling2D(conv3, "pool3")
        norm3 = LRN2D(pool3, "norm3")

        conv4 = Conv2D(norm2, "conv4")
        pool4 = MaxPooling2D(conv4, "pool4")
        norm4 = LRN2D(pool4, "norm4")

        dense0 = Flatten(norm4)
        dense1 = Dense(dense0, 64, "dense1")
        output = Dense(dense1, 128, "dense2")

    softmax = Softmax(output, n_classes, "softmax")
    return softmax


def inference(images, batch_size, n_classes, setting="baseline"):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    #conv1, shape = [kernel size, kernel size, channels, kernel numbers]

    assert len(model_list) == len(parameters)
    assert setting in model_list
    print ("Total parameters: %s" % str(parameters[model_list.index(setting)]))

    logits = get_model(images, batch_size, n_classes, model_list.index(setting))
    return logits

def loss(logits, labels):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]

    Returns:
        loss tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss


def training(loss, learning_rate, global_step):
    '''Training ops, the Op returned by this function is what must be passed to
        'sess.run()' call to cause the model to train.

    Args:
        loss: loss tensor, from losses()

    Returns:
        train_op: The op for training
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        # global_step = tf.Variable(0)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).
    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
    """
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy', accuracy)
    return accuracy

if __name__ == "__main__":
    for setting in model_list[0:5]: # [6:9]:
        with tf.Graph().as_default():
            print ("model: " + setting)
            inputs = tf.zeros((128, 208, 208, 3))
            outputs = inference(inputs, 128, 2, setting=setting)
            print (outputs)
