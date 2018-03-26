from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from utils import LRN2D

nb_classes = 2
img_rows, img_cols = 208, 208
batch_size = 128

model = Sequential()
model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same',
                 activation='relu', input_shape=(208,208,3)))
model.add(MaxPooling2D((3,3), (2, 2), padding='same'))
# model.add(BatchNormalization())
model.add(LRN2D())

'''
model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same',
                 activation='relu'))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
# model.add(BatchNormalization())
model.add(LRN2D())
'''

model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same',
                 activation='relu'))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(LRN2D())
# model.add(BatchNormalization())

model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same',
                 activation='relu'))
model.add(MaxPooling2D((3, 3), strides=(1, 1), padding='same'))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))

model.summary()

# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])
