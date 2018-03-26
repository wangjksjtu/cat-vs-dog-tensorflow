import numpy as np
import h5py
from scipy import ndimage, misc
import glob
import os

num_list = [3, 4] # range(1, 5) # range(5, 11) # [15, 20, 25, 50, 75]
# print num_list
img_dirs = ["quality_" + str(i) for i in num_list]

def load_data(img_dirs, image_W, image_H):
    images = []
    labels = []

    files = glob.glob(os.path.join(img_dirs, "*.jpg"))

    for i, filepath in enumerate(files):
        name = filepath.split('/')[-1].split(".")
        if name[0]=='cat':
            labels.append(0)
        else:
            labels.append(1)

        image = ndimage.imread(filepath, mode="RGB")
        image_resized = misc.imresize(image, (image_W, image_H))
        images.append(image_resized / 255.0)
        # if i == 999:
        #    break

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

    for img_dir in img_dirs:
        data, label = load_data(img_dir, 208, 208)
        save_h5(data, label, os.path.join(img_dir, "data.h5"))
        # data, label = read_h5(os.path.join(img_dir, "data.h5"))
        # print label.value
        print img_dir
    # print data[0,0,:], label[0]
