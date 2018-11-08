"""Build and save models 1 and 2"""

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers import Flatten, Dense, Input
from keras.models import Model
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

batch_size = 16

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
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=True
)
val_gen = test_datagen.flow_from_directory(
    'images/val',
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=False
)

train_steps = train_gen.samples // batch_size
val_steps = val_gen.samples // batch_size
num_classes = 83
inp = Input(shape=(224, 224, 3))

# Build and train model 1
base_model = ResNet50(input_tensor=inp, weights='imagenet', include_top=False, classes=num_classes)
last_layer = base_model.output

for layer in base_model.layers:
    layer.trainable = False

# Add new flatten and dense layers
last_layer = Flatten(name='flatten_2')(last_layer)
out = Dense(num_classes, activation='softmax', name='output_layer')(last_layer)

model = Model(inputs=inp, outputs=out)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit_generator(
    train_gen,
    steps_per_epoch=train_steps,
    epochs=2,
    verbose=1,
    validation_data=val_gen,
    validation_steps=val_steps
)
model.save('resnet50_freeze.h5')

val_gen = test_datagen.flow_from_directory(
    'images/val',
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=False
)
predictions = model.predict_generator(val_gen, val_steps)

y_pred = np.argmax(predictions, axis=1)
y_true = val_gen.classes[0:len(y_pred)]

print(precision_recall_fscore_support(y_true, y_pred, pos_label=None, average='macro'))
print(accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))

# Build and train model 2
base_model = ResNet50(input_tensor=inp, weights='imagenet', include_top=False, classes=num_classes)
last_layer = base_model.output

# Add new flatten and dense layers
last_layer = Flatten(name='flatten_2')(last_layer)
out = Dense(num_classes, activation='softmax', name='output_layer')(last_layer)

model = Model(inputs=inp, outputs=out)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit_generator(
    train_gen,
    steps_per_epoch=train_steps,
    epochs=2,
    verbose=1,
    validation_data=val_gen,
    validation_steps=val_steps
)
model.save('resnet50.h5')

val_gen = test_datagen.flow_from_directory(
    'images/val',
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=False
)
predictions = model.predict_generator(val_gen, val_steps)

y_pred = np.argmax(predictions, axis=1)
y_true = val_gen.classes[0:len(y_pred)]

print(precision_recall_fscore_support(y_true, y_pred, pos_label=None, average='macro'))
print(accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))
