from getdata import get_data, get_user_file
import os
from printimages import plot_accuracy, show_dataset_examples, recognize_user_images, print_images
import tensorflow as tf
import numpy as np
from keras_preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from get_keras_model import get_new_model, train_model, get_pre_trained_model
from easygui import fileopenbox, filesavebox

"""
This file executes script that trains neural network
to determine objects of any number of given types

Types are defined from given input directory
that contains folders named respectively.

E.g. to work with dataset rock-paper-scissors the given directory
must contain folders:
rocks - with training pictures of rocks
papers - with training pictures of papers
scissors - with training pictures of scissors

Same folder structure must be provided for validation data as well
"""

# input parameters for different workflows
SHOW_DATASET_EXAMPLE = False
USE_TEST_FOLDERS = False
TARGET_SIZE = 200
USE_INCEPTION_MODEL = True
AUGMENT = True
is_plot_accuracy = True # at the meantime accuracy plotted only for new models
# TODO: use CSVLogger to store statistics along with model
#

if USE_TEST_FOLDERS:  # for saving time during tests - give hardcoded folders
    train_dirs = []
    validation_dirs = []
    labels = []
else:
    labels, train_dirs, validation_dirs = get_data()

VALIDATE = (len(validation_dirs) == len(train_dirs))  # do not validate if validation set is not provided

number_of_classes = len(labels)
work_dir = os.path.dirname(train_dirs[0])
print('Training dir: ', work_dir)
work_dir_validation = None
if VALIDATE:
    work_dir_validation = os.path.dirname(validation_dirs[0])
    print('Validation dir ', work_dir_validation)

if SHOW_DATASET_EXAMPLE:
    show_dataset_examples(train_dirs, 'Random train images', number_of_images=5)

# use saved trained model
print('Load existing trained model')
model_path = fileopenbox(
    title='Load keras model for: {}'.format(', '.join(labels)),
    filetypes=[['*.h5', 'Keras Model']])

if model_path is not None:
    model = tf.keras.models.load_model(model_path)
    is_plot_accuracy = False
else:
    # define TensorFlow model - refactored into function to be able to use pre-trained model
    if USE_INCEPTION_MODEL:
        model = get_pre_trained_model(TARGET_SIZE, TARGET_SIZE, number_of_classes)
    else:
        model = get_new_model(TARGET_SIZE, TARGET_SIZE, number_of_classes)

    # all images will be rescaled by 1.0 / 255
    # add augmentation to increase dataset size
    if AUGMENT:
        train_datagen = ImageDataGenerator(
            rescale=1 / 255.0,
            rotation_range=40,  # random rotation from 0 to 40 degrees
            width_shift_range=0.2,  # random shift width-wise from 0 to 0.2
            height_shift_range=0.2,  # random shift height-wise
            shear_range=0.2,  # shear - i.e. transform image
            zoom_range=0.2,  # random zooming
            horizontal_flip=True,
            fill_mode='nearest'  # how to fill lost pixels
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1 / 255.0)

    validation_datagen = ImageDataGenerator(rescale=1 / 255.0)

    if len(labels) > 2:
        class_mode = 'categorical'
    else:
        class_mode = 'binary'

    # flow training images in batches of 20
    train_generator = train_datagen.flow_from_directory(
        work_dir,  # directory with all images
        target_size=(TARGET_SIZE, TARGET_SIZE),  # all images will be resized to ...
        batch_size=16,
        class_mode=class_mode  # binary labels
    )
    validation_generator = None
    if VALIDATE:
        validation_generator = validation_datagen.flow_from_directory(
            work_dir_validation,
            target_size=(TARGET_SIZE, TARGET_SIZE),
            batch_size=16,
            class_mode=class_mode
        )

    # training process refactored into function
    history = train_model(model,
                          train_generator=train_generator,
                          validation_generator=validation_generator)

    print('\nSave trained model:')
    save_path = filesavebox(msg='Save model', filetypes='*.h5',
                            default='{}.h5'.format('_'.join(labels)).lower())
    if save_path:
        model.save(save_path)
        print('Model saved: ', save_path)

# printIntermediateRepresentations(show_images, model)

# plot how the accuracy evolves during the training
if is_plot_accuracy:
    plot_accuracy(history, VALIDATE)

recognize_user_images(model, labels, TARGET_SIZE, TARGET_SIZE)

