import numpy as np
import h5py
from scipy import ndimage, misc
import glob
import os

num_list = range(1, 11) +  [15, 20, 25, 50, 75] # range(1, 5) # range(5, 11) # [15, 20, 25, 50, 75]
# print num_list
img_dirs = ["quality_" + str(i) for i in num_list]

def get_labels(img_dir, filename):
    f = open(os.path.join(img_dir, filename), "r")
    labels = f.readlines()
    for i, label in enumerate(labels):
        labels[i] = label.rstrip()
    return labels

label_list = get_labels(img_dirs[0], 'wnids.txt')

def load_data(img_dirs, label_list): #image_W, image_H):
    images = []
    labels = []
    for i, label in enumerate(label_list):
        img_dir = os.path.join(img_dirs, "val/" + label + "/images")
        files = glob.glob(os.path.join(img_dir, "*.JPEG"))
        for j, filepath in enumerate(files):
            labels.append(i)
            image = ndimage.imread(filepath, mode="RGB")
            # image_resized = misc.imresize(image, (image_W, image_H)) / 255.0
            image_resized = image / 255.0
            image_float32 = image_resized.astype('float32')
            images.append(image_float32)
            # if j == 20:
            #     break
    images = np.stack(images, axis = 0)
    labels = np.asarray(labels)
    print images.shape, labels.shape
    return images, labels

def save_h5(data, label, filename):
    f = h5py.File(filename,'w')
    f['data'] = data
    f['label'] = label
    f.close()

def read_h5(filename):
    f = h5py.File(filename,'r')
    data, label = f['data'], f['label']
    return data, label


if __name__ == "__main__":
    print img_dirs
    # load_data(img_dirs[0], label_list)
    for img_dir in img_dirs:
        data, label = load_data(img_dir, label_list)
        save_h5(data, label, os.path.join(img_dir, "data_val.h5"))
        data, label = read_h5(os.path.join(img_dir, "data_val.h5"))
        # print data.value.shape, label.value.shape
        print img_dir
    # print data[0,0,:], label[0]
