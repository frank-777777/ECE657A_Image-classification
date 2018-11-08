"""
Implements a 14-layer residual network similar to resnet18 for training from scratch.
"""

import keras
import numpy as np
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    ZeroPadding2D
)
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dense, Input, Flatten
from keras.models import Model
from keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img
)
from keras.preprocessing import image
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
from keras import backend as K

def conv_block(input_tensor, filters, stage, block, strides=(2, 2)):
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters, (3, 3), strides=strides,
               name=conv_name_base + '2a', padding="same")(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), name=conv_name_base + '2b', padding="same")(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    shortcut = Conv2D(
        filters,
        (1, 1),
        strides=strides,
        padding="valid",
        name=conv_name_base + '1'
    )(input_tensor)

    x = keras.layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def identity_block(input_tensor, filters, stage, block):
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters, (3, 3), name=conv_name_base + '2a', padding="same")(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), name=conv_name_base + '2b', padding="same")(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    x = keras.layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def resnet14(input_tensor, n_classes):
    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_tensor)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)

    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 64, stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 64, stage=2, block='b')

    x = conv_block(x, 128, stage=3, block='a')
    x = identity_block(x, 128, stage=3, block='b')

    x = conv_block(x, 256, stage=4, block='a')
    x = identity_block(x, 256, stage=4, block='b')

    x = AveragePooling2D((5, 5), name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(n_classes, activation='softmax', name='dense')(x)

    return Model(input_tensor, x, name='resnet34')


batch_size = 128

train_datagen = ImageDataGenerator(
    rotation_range=360,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.35,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator()

train_gen = train_datagen.flow_from_directory(
    'images/train',
    target_size=(150, 150),
    batch_size=batch_size,
    color_mode='grayscale',
    shuffle=True
)
val_gen = test_datagen.flow_from_directory(
    'images/val',
    target_size=(150, 150),
    batch_size=batch_size,
    color_mode='grayscale',
    shuffle=False
)

train_steps = train_gen.samples // batch_size
val_steps = val_gen.samples // batch_size

num_classes = 83
inp = Input(shape=(150, 150, 1))
model = resnet14(inp, num_classes)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit_generator(
    train_gen,
    steps_per_epoch=train_steps,
    epochs=12,
    verbose=1,
    validation_data=val_gen,
    validation_steps=val_steps
)
model.save('resnet14.h5')

val_gen = test_datagen.flow_from_directory(
    'images/val',
    target_size=(150, 150),
    batch_size=batch_size,
    shuffle=False,
    color_mode='grayscale'
)
predictions = model.predict_generator(val_gen, val_steps)

y_pred = np.argmax(predictions, axis=1)
y_true = val_gen.classes[0:len(y_pred)]

print(precision_recall_fscore_support(y_true, y_pred, pos_label=None, average='macro'))
print(accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))
