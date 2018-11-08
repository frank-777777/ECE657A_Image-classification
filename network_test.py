"""Code to test a trained model on the test set"""

import numpy as np
import keras.models
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Load a trained model
model = keras.models.load_model('resnet14.h5')
batch_size = 64
image_size = (150, 150)
color_mode = "grayscale"

# Test the loaded model
test_datagen = ImageDataGenerator()

test_gen = test_datagen.flow_from_directory(
    'images/test',
    target_size=image_size,
    batch_size=batch_size,
    color_mode=color_mode,
    shuffle=False
)

test_steps = test_gen.samples // batch_size + 1

predictions = model.predict_generator(test_gen, test_steps)

y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes[0:len(y_pred)]

print(precision_recall_fscore_support(y_true, y_pred, pos_label=None, average='macro'))
print(accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))