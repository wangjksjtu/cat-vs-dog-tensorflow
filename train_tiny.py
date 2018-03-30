import os
import numpy as np
import tensorflow as tf
import provider
import model_tiny as model
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--quality', type=int, default=5, help='Image quality [default: 0]')
parser.add_argument('--setting', type=int, default=9, help='Model architecture (0-9) [default: 9]')
parser.add_argument('--model', type=str, default="", help='Model architecture description (0-9) [default: ""]')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size during training [default: 512]')
parser.add_argument('--num_epoch', type=int, default=1000, help='Batch size during training [default: 1000]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.0001]')
# parser.add_argument('--max_step', type=int, default=20000, help='Max training step [default: 0.0001]')
FLAGS = parser.parse_args()

# MAX_STEP = FLAGS.max_step
MAX_EPOCH = FLAGS.num_epoch
learning_rate = FLAGS.learning_rate
BATCH_SIZE = FLAGS.batch_size
QUALITY = FLAGS.quality
SETTING = FLAGS.setting
MODEL = FLAGS.model

model_list = ["io-units_x1+cnn_pooling2",
              "io-units_x1+cnn_pooling",
              "io-units_x2",
              "io-layer",
              "io-units_x1",
              "io-units_x1+layer",
              "io-units_x1+units_x1",
              "io-units_x1+units_x2",
              "baseline",
              "original"
             ]

if MODEL == "":
    SETTING = model_list[SETTING]
else:
    SETTING = MODEL

print ("model: " + SETTING)

N_CLASSES = 200
IMG_W = 64  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 64
RATIO = 0.2 # take 20% of dataset as validation data
# BATCH_SIZE = 128
CAPACITY = 20000
# MAX_STEP = 20000 # with current parameters, it is suggested to use MAX_STEP>10k
# learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001

train_dir = "data/quality_" + str(QUALITY) + "/"
# val_dir = train_dir
logs_dir = "logs/" + str(SETTING) + "/quality_" + str(QUALITY) + "/"
logs_train_dir = "logs/" + str(SETTING) + "/quality_" + str(QUALITY) + "/" + "train/"
logs_val_dir = "logs/" + str(SETTING) + "/quality_" + str(QUALITY) + "/" + "val/"

print ("train_dir: " + train_dir)
# print ("val_dir": + val_dir)
print ("logs_train_dir: " + logs_train_dir)
print ("logs_val_dir: " + logs_val_dir)

assert (os.path.exists(train_dir))
if not os.path.exists(logs_train_dir):
    os.makedirs(logs_train_dir)
if not os.path.exists(logs_val_dir):
    os.makedirs(logs_val_dir)

os.system('cp %s %s' % ("model_paras.py", logs_dir)) # bkp of model def
os.system('cp train.py %s' % (logs_dir)) # bkp of train procedure

LOG_FOUT = open(os.path.join(logs_train_dir, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

# train_dir = 'data/train/'
# logs_train_dir = 'logs/original/train/'
# logs_val_dir = 'logs/orginal/val/'

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def run_training(train_dir, logs_train_dir, logs_val_dir):

    # train, train_label, val, val_label = provider.get_files(train_dir, RATIO)
    '''
    train_batch, train_label_batch = provider.get_batch(train,
                                                  train_label,
                                                  IMG_W,
                                                  IMG_H,
                                                  BATCH_SIZE,
                                                  CAPACITY)
    val_batch, val_label_batch = provider.get_batch(val,
                                                  val_label,
                                                  IMG_W,
                                                  IMG_H,
                                                  BATCH_SIZE,
                                                  CAPACITY)
    '''
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    batch = tf.Variable(0, name='global_step', trainable=False)

    logits = model.inference(x, BATCH_SIZE, N_CLASSES, setting=SETTING)
    print logits
    loss = model.loss(logits, y_)
    print loss
    # acc = model.evaluation(logits, y_)
    train_op = model.training(loss, learning_rate, batch)

    print "[Loading data ...]"
    data, label = provider.load_data(train_dir)
    data, label, _ = provider.shuffle_data(data, label)
    print "[Finish Loading]"

    file_size = len(label)
    val_data, val_label = data[int((1-RATIO)*file_size):, ...], label[int((1-RATIO)*file_size):]
    train_data, train_label = data[0:int((1-RATIO)*file_size), ...], label[0:int((1-RATIO)*file_size)]

    # train_data, train_label = provider.load_data(train_dir)
    # train_data, train_label, _ = provider.shuffle_data(train_data, train_label)
    # val_data, val_label = provider.load_data(val_dir)
    # val_data, val_label, _ = provider.shuffle_data(val_data, val_label)
    print "[Finish Loading]"

    print ("train_data", train_data.shape, train_label.shape)
    print ("val_data", val_data.shape, val_label.shape)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)

        ops = {'images_pl': x,
               'labels_pl': y_,
               'logits': logits,
               'loss': loss,
               'train_op': train_op,
               'merged': summary_op,
               'step': batch}

        try:
            for epoch in np.arange(MAX_EPOCH):
                if coord.should_stop():
                        break

                log_string('**** EPOCH %03d ****' % (epoch))
                sys.stdout.flush()

                train_one_epoch(sess, ops, train_writer, train_data, train_label)
                eval_one_epoch(sess, ops, val_writer, val_data, val_label)

                # Save the variables to disk.
                if epoch % 10 == 0:
                    save_path = saver.save(sess, os.path.join(logs_dir, "model.ckpt"))
                    log_string("Model saved in file: %s" % save_path)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

def train_one_epoch(sess, ops, train_writer, current_data, current_label):
    """ ops: dict mapping from string to tf ops """
    # current_data, current_label = provider.load_data(train_dir)
    # file_size = len(current_label)
    # current_data, current_label = current_data[0:int((1-RATIO)*file_size), ...],  current_label[0:int((1-RATIO)*file_size)]
    # current_data, current_label, _ = provider.shuffle_data(current_data, current_label)
    # print _
    # print (current_data.shape, current_label.shape)

    train_size = len(current_label)
    num_batches = train_size // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['images_pl']: current_data[start_idx:end_idx,...],
                        ops['labels_pl']: current_label[start_idx:end_idx]}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['logits']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        # print pred_val, loss_val
        pred_val = np.argmax(pred_val, 1)
        # print pred_val, current_label[start_idx:end_idx]
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        # print correct
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val
        # print (total_correct / float(total_seen))

    # print (loss_sum, num_batches)
    # print (total_correct, total_seen)
    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))

def eval_one_epoch(sess, ops, val_writer, current_data, current_label):
    """ ops: dict mapping from string to tf ops """
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(N_CLASSES)]
    total_correct_class = [0 for _ in range(N_CLASSES)]

    # current_data, current_label = provider.load_data(train_dir)
    # file_size = len(current_label)
    # current_data, current_label = current_data[int((1-RATIO)*file_size):, ...],  current_label[int((1-RATIO)*file_size):]
    # current_data, current_label, _ = provider.shuffle_data(current_data, current_label)
    # print _
    # print (current_data.shape, current_label.shape)

    val_size = current_label.shape[0]
    num_batches = val_size // BATCH_SIZE

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['images_pl']: current_data[start_idx:end_idx, ...],
                        ops['labels_pl']: current_label[start_idx:end_idx]}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['logits']], feed_dict=feed_dict)
        val_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            l = current_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i-start_idx] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class) / np.array(total_seen_class,dtype=np.float))))

if __name__ == "__main__":
    run_training(train_dir, logs_train_dir, logs_val_dir)
