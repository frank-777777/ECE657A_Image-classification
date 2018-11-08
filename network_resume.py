"""Continuing training a model already saved to a file"""

import numpy as np
import keras.models
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Load a trained model
model_name = 'resnet50.h5'
model = keras.models.load_model(model_name)
batch_size = 64
image_size = (224, 224)
color_mode = "rgb"


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
    target_size=image_size,
    batch_size=batch_size,
    color_mode=color_mode,
    shuffle=True
)
val_gen = test_datagen.flow_from_directory(
    'images/val',
    target_size=image_size,
    batch_size=batch_size,
    color_mode=color_mode,
    shuffle=False
)

train_steps = train_gen.samples // batch_size
val_steps = val_gen.samples // batch_size

model.fit_generator(
   train_gen,
   steps_per_epoch=train_steps,
   epochs=1,
   verbose=1,
   validation_data=val_gen,
   validation_steps=val_steps
)
model.save(model_name)

val_gen = test_datagen.flow_from_directory(
    'images/val',
    target_size=image_size,
    batch_size=batch_size,
    color_mode=color_mode,
    shuffle=False
)
predictions = model.predict_generator(test_gen, test_steps)

y_pred = np.argmax(predictions, axis=1)
y_true = val_gen.classes[0:len(y_pred)]

print(precision_recall_fscore_support(y_true, y_pred, pos_label=None, average='macro'))
print(accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))
