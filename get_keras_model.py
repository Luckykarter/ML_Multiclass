import tensorflow as tf
import concurrent.futures  # fire learning in separate thread to be able to stop it manually
from stoptraining import ManualStop
from numba import jit, cuda
import requests
import os
from tensorflow.keras.applications.inception_v3 import InceptionV3

def _get_funcs(number_of_classes):
    # support for multi-class definitions - difference in the last layer
    if number_of_classes > 2:
        loss = tf.keras.losses.categorical_crossentropy
        number_of_neurons = number_of_classes
        activation = tf.keras.activations.softmax
    else:
        loss = tf.keras.losses.binary_crossentropy
        number_of_neurons = 1
        activation = tf.keras.activations.sigmoid
    return loss, number_of_neurons, activation

def get_pre_trained_model(image_width, image_height, number_of_classes):
    local_weights_file = 'resources/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    if not os.path.exists(local_weights_file):
        print('Downloading file from URL')
        url = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        try:
            r = requests.get(url, allow_redirects=True)
            open(local_weights_file, 'wb').write(r.content)
        except Exception as e:
            print('Unable to get model')
            print(e)
            exit(1)
        if os.path.exists(local_weights_file) and os.path.getsize(local_weights_file) > 0:
            print('Model downloaded successfully')
    pre_trained_model = InceptionV3(
        input_shape=(image_width, image_height, 3),
        include_top=False,
        weights=None
    )

    pre_trained_model.load_weights(local_weights_file)
    for layer in pre_trained_model.layers:
        # print(layer.name, layer.output_shape)
        layer.trainable = False
    last_layer = pre_trained_model.get_layer('mixed7')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    # flatten the ouput layer to 1 dimension
    x = tf.keras.layers.Flatten()(last_output)

    # add a fully connected layer with 1024 hidden units
    x = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(x)

    # add a dropout
    x = tf.keras.layers.Dropout(0.2)(x)

    # for multi-class we need different parameters
    loss, number_of_neurons, activation = _get_funcs(number_of_classes)

    # add a final sigmoid layer for classification
    x = tf.keras.layers.Dense(number_of_neurons, activation=activation)(x)
    model = tf.keras.Model(pre_trained_model.input, x)

    model.compile(loss=loss,
                  # optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def get_new_model(image_width, image_height, number_of_classes):
    # for multi-class we need different parameters
    loss, number_of_neurons, activation = _get_funcs(number_of_classes)
    model = tf.keras.models.Sequential([
        # 1
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, input_shape=(image_width, image_height, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # 2
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2, 2),
        # 3
        tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2, 2),
        # 4 and 5 are removed because size of pictures decreased
         tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu),
         tf.keras.layers.MaxPooling2D(2, 2),
        # 5
        #  tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu),
        #  tf.keras.layers.MaxPooling2D(2, 2),
        # flatten the image
        tf.keras.layers.Flatten(),
        #Dropout
        tf.keras.layers.Dropout(0.5),
        # 512 connected neurons
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        # 0 - horses, 1 - humans
        tf.keras.layers.Dense(number_of_neurons, activation=activation)
    ])
    model.compile(loss=loss,
                  optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
                  metrics=['accuracy'])
    model.summary()
    return model


# training:
def _train(model, train_generator, validation_generator):
    return model.fit(train_generator,
                     steps_per_epoch=16, # 16
                     epochs=25, #40
                     callbacks=None,
                     verbose=1,
                     validation_data=validation_generator,
                     validation_steps=8
                     )


def _concur_train(model, train_generator, validation_generator):
    with concurrent.futures.ThreadPoolExecutor() as e:
        ms = ManualStop(model)
        e.submit(ms.start_listener)
        train = e.submit(_train, model, train_generator, validation_generator)
        res = train.result()
        if ms.running:
            ms.stop_listener() # stop listen to keys when training is finished
        return res

# Try to use videocard for processing
@jit(target='cuda')
def _train_with_gpu(model, train_generator, validation_generator):
    print('Start training using GPU')
    return _concur_train(model, train_generator, validation_generator)


def train_model(model, train_generator, validation_generator):
    try:
        return _train_with_gpu(model, train_generator, validation_generator)
    except Exception as e:
        print(e)
        print('Start training using CPU')
        return _concur_train(model, train_generator, validation_generator)
