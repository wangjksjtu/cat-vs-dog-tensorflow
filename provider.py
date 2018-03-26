import tensorflow as tf
import numpy as np
import os
import math
import glob
from scipy import ndimage, misc
from data.prepare import read_h5

# you need to change this to your data directory
train_dir = 'data/train/'

def get_files(file_dir, ratio):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in glob.glob(os.path.join(file_dir, "*.jpg")):
        name = file.split('/')[-1].split(".")
        if name[0]=='cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))

    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    all_image_list = temp[:, 0]
    all_label_list = temp[:, 1]

    n_sample = len(all_label_list)
    n_val = math.ceil(n_sample*ratio) # number of validation samples
    n_train = n_sample - n_val # number of trainning samples
    n_val, n_train = int(n_val), int(n_train)
    train_images = all_image_list[0:n_train]
    train_labels = all_label_list[0:n_train]
    train_labels = [int(float(i)) for i in train_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]



    return train_images, train_labels, val_images, val_labels

def load_data(img_dirs):
    f = os.path.join(img_dirs, "data.h5")
    data, label = read_h5(f)
    return data.value, label.value

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
            data: B,... numpy array
            label: B, numpy array
        Return:
            shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    print (image)
    ######################################
    # data argumentation should go to here
    ######################################

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # if you want to test the generated batches of images, you might want to comment the following line.
    print (image)

    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64,
                                                capacity = capacity)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch


def test_generated_batch():
    import matplotlib.pyplot as plt

    BATCH_SIZE = 2
    CAPACITY = 256
    IMG_W = 208
    IMG_H = 208

    train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'
    ratio = 0.2
    train_images, train_labels, val_images, val_labels = get_files(train_dir, ratio)
    train_image_batch, train_label_batch = get_batch(train_images, train_labels, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop() and i<1:

                img, label = sess.run([train_image_batch, train_label_batch])
                # img, label = sess.run([val_image, val_label_batch])

                # just test one batch
                for j in np.arange(BATCH_SIZE):
                    print('label: %d' %label[j])
                    plt.imshow(img[j,:,:,:])
                    plt.show()
                i+=1

        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    f = os.path.join("data/quality_0", "data.h5")
    data, label = read_h5(f)
    print (data.value.shape, label.value.shape)

