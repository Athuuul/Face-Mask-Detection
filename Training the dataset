from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import re
import os

# Check if the file exists, and if so, delete it
model_file = 'model.keras'
if os.path.exists(model_file):
    os.remove(model_file)

num_classes = 2
model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
model.add(Dense(num_classes, activation='softmax'))

# not using the first layer for training
model.layers[0].trainable = False

model.compile(optimizer='sgd', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])

# data augmentation for training images
data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                              horizontal_flip=True, 
                                              width_shift_range=0.1,
                                              height_shift_range=0.1)
            
# Specify no augmentation that will be used for validation data
data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

# Take image directly from directory and apply augmentation
image_size = 224

# Prepare training images
train_generator = data_generator_with_aug.flow_from_directory(
                                        directory='input/mask-detection/images/train',
                                        classes=['with_mask', 'without_mask'],
                                        target_size=(image_size, image_size),
                                        batch_size=10,
                                        class_mode='categorical')

# Prepare validation images
validation_generator = data_generator_no_aug.flow_from_directory(
                                        directory='input/mask-detection/images/val',
                                        classes=['with_mask', 'without_mask'],
                                        target_size=(image_size, image_size),
                                        class_mode='categorical')

# Train the model
fit_stats = model.fit(train_generator,
                      steps_per_epoch=60,
                      epochs=4,
                      validation_data=validation_generator,
                      validation_steps=1)

# Testing with test images
test_generator = data_generator_no_aug.flow_from_directory(
    directory='input/mask-detection/images/test',
    target_size=(image_size, image_size),
    batch_size=10,
    class_mode=None,
    shuffle=False
)

# Predict from generator (returns probabilities)
pred = model.predict(test_generator, steps=len(test_generator), verbose=1)
cl = np.round(pred)
filenames = test_generator.filenames

# Extract real class from filenames
real_class = []
for file in filenames:
    if re.search("with_mask", file):
        real_class.append(1.0)
    else:
        real_class.append(0.0)

# Extract predicted class from probabilities
predicted_class = cl[:, 0]

# Calculate accuracy
accuracy = sum(1 for x, y in zip(real_class, predicted_class) if x == y) / len(real_class)
print("Accuracy:", accuracy)

# Save the model
model.save('model.keras')
