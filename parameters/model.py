import argparse
import os
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from utils import LRN2D

parser = argparse.ArgumentParser()
parser.add_argument('--setting', type=int, default=8, help='Model architecture (0-8) [default: 8]')
parser.add_argument('--model', type=str, default="", help='Model architecture description (0-8) [default: ""]')
parser.add_argument('--all', type=bool, default=False, help="Print all architecures [default: False]")
parser.add_argument('--save_dir', type=str, default="models", help="The output directory path [default: models/]")
FLAGS = parser.parse_args()

nb_classes = 2
img_rows, img_cols = 208, 208

SETTING = FLAGS.setting
MODEL = FLAGS.model
ALL = FLAGS.all
SAVE_DIR = FLAGS.save_dir

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

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

if MODEL == "":
    SETTING = model_list[SETTING]
else:
    SETTING = MODEL

setting = model_list.index(SETTING)

def build_model(setting):
    print ("[model]: " + model_list[setting])

    model = Sequential()
    model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', input_shape=(208,208,3)))
    model.add(MaxPooling2D((3,3), (2, 2), padding='same'))
    # model.add(BatchNormalization())
    model.add(LRN2D())

    model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same',
                     activation='relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    # model.add(BatchNormalization())
    model.add(LRN2D())
    if setting == 9:
        # [orginal model]
        model.pop()
        model.pop()
        model.add(MaxPooling2D((3, 3), strides=(1, 1), padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
    elif setting == 8:
        # [baseline model]
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
    elif setting == 7:
        # [io-units_x1+units_x2 model]
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(512, activation='relu'))
    elif setting == 6:
        # [io-units_x1+units_x1 model]
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(256, activation='relu'))
    elif setting == 5:
        # [io-units_x1+layer model]
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
    elif setting == 4:
        # [io-units_x1 model]
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(128, activation='relu'))
    elif setting == 3:
        # [io-layer model]
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
    elif setting == 2:
        # [io-units_x2]
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(128, activation='relu'))
    elif setting == 1:
        # [io-units_x1+cnn_pooling]
        model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        # model.add(BatchNormalization())
        model.add(LRN2D())

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(128, activation='relu'))
    else:
        model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        # model.add(BatchNormalization())
        model.add(LRN2D())

        model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        # model.add(BatchNormalization())
        model.add(LRN2D())

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(128, activation='relu'))

    model.add(Dense(nb_classes, activation='softmax'))
    return model

def save_summary(model, header, suffix):
    assert(suffix.split(".")[0] == "")
    with open(file_header + suffix,'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


from keras.utils import plot_model #, print_summary
if ALL:
    for setting in range(len(model_list)):
        model = build_model(setting)
        description = model_list[setting]
        file_header = os.path.join(SAVE_DIR, "model_" + str(model.count_params()) + "_" + \
                                description)
        plot_model(model, to_file=file_header+".pdf", show_shapes=True)
        save_summary(model, file_header, ".txt")
else:
    model = build_model(setting)
    file_header = os.path.join(SAVE_DIR, "model_" + str(model.count_params()) + "_" + \
                             SETTING)
    plot_model(model, to_file=file_header + ".pdf", show_shapes=True)
    save_summary(model, file_header, ".txt")

# model.summary()


# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
