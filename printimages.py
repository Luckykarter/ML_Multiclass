from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from math import sqrt, ceil
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
import os
from easygui import fileopenbox
from keras_preprocessing import image


# prints given images in automaticaly defined grid XnY
def print_images(images, title, img_titles=None):
    ncols = ceil(sqrt(len(images)))
    nrows = ceil(len(images) / ncols)

    # fig = plt.gcf()
    fig = plt.figure()
    fig.canvas.set_window_title(title)
    # fig.set_size_inches(ncols * 4, nrows * 4)


    for i, img_path in enumerate(images):
        # set up subplot
        sp = plt.subplot(nrows, ncols, i + 1)
        if img_titles:
            sp.set_title(img_titles[i])

        sp.axis('Off')

        img = mpimg.imread(img_path)
        plt.imshow(img)
    plt.show()
    plt.close()


def print_intermediate_representations(images, model):
    successive_outputs = [layer.output for layer in model.layers[1:]]
    visualization_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=successive_outputs
    )
    img_path = random.choice(images)

    img = load_img(img_path, target_size=(300, 300))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    x /= 255

    successive_feature_maps = visualization_model.predict(x)
    layer_names = [layer.name for layer in model.layers]

    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        # do this for conv/maxpool layers but not the fully-connected layers
        if len(feature_map.shape) == 4:
            n_features = feature_map.shape[-1]  # number of features in feature map
            # the feature map has shape (1, size, size, n_features)

            size = feature_map.shape[1]
            display_grid = np.zeros((size, size * n_features))
            for i in range(n_features):
                x = feature_map[0, :, :, i]
                x -= x.mean()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                display_grid[:, i * size:(i + 1) * size] = x
            scale = 20.0 / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid)
    plt.show()


# opens separate windows with random examples of training and validation images
def show_dataset_examples(dirs, label, number_of_images=10):
    show_images = []
    for dir_ in dirs:
        if dir_:
            names = os.listdir(dir_)
            show_images += [os.path.join(dir_, name)
                            for name in random.choices(names, k=number_of_images)]
    if show_images:
        print_images(show_images, label)


def plot_accuracy(history, validate=False):
    accuracy = history.history['accuracy']
    loss = history.history['loss']
    val_accuracy, val_loss = None, None
    title = 'Training '

    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'r', label='Accuracy')
    if validate:
        title += 'and Validation '
        val_accuracy = history.history['val_accuracy']
        val_loss = history.history['val_loss']
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title(title + 'accuracy')
    plt.legend()
    plt.grid()
    plt.show()
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Loss')
    if validate:
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(title + 'loss')
    plt.legend()
    plt.grid()
    plt.show()


def recognize_user_images(model, labels, width, height):
    print('Recognize user images: ')

    while True:
        img_paths = fileopenbox(
            msg="Types: {}".format(', '.join(labels)),
            title="Choose image(s)",
            multiple=True
        )
        if not img_paths:
            break
        print_labels = []
        for file in img_paths:

            try:
                img = image.load_img(
                    file, target_size=(width, height)
                )

                x = image.img_to_array(img) * (1 / 255.0)
                x = np.expand_dims(x, axis=0)
                images = np.vstack([x])
                classes = model.predict(images, batch_size=10)
                if len(labels) <= 2:
                    if classes[0] > 0.5:
                        print_labels.append(labels[0])
                    else:
                        print_labels.append(labels[1])
                else:
                    # to get one class directly:
                    # y_cl = classes.argmax(axis=-1)
                    # print(labels[y_cl[0]])
                    results = dict()
                    for i in range(len(labels)):
                        results[labels[i]] = round(classes[0][i]*100, 2)
                    # print(file, results)

                    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
                    top_res = results[0]
                    guess = top_res[0] + ' ({}%)'.format(top_res[1])
                    if top_res[1] < 60:
                        guess += '\n{} ({}%)'.format(results[1][0], results[1][1])

                    print_labels.append(guess)

            except Exception as e:  # it is not an image - skip it
                print(e)
                continue
        if len(img_paths) == len(print_labels):
            print_images(img_paths, 'Neural network guesses', print_labels)