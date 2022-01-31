import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import shift
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


def read_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            vector = line.split()
            numerical_vector = list(map(lambda x: int(x), vector))
            data.append(numerical_vector)

    labels = [i for i in range(0, 10) for _ in range(200)]
    return data, labels


def normalize(X):
    return X / 6


def split_data(features, labels):
    train_features, test_features, train_labels, test_labels = [], [], [], []
    for i in range(0, len(features), 200):
        train_features.extend(features[i : i + 100])
        test_features.extend(features[i + 100 : i + 200])

        train_labels.extend(labels[i : i + 100])
        test_labels.extend(labels[i + 100 : i + 200])

    return (
        normalize(np.array(train_features)),
        np.array(train_labels),
        normalize(np.array(test_features)),
        np.array(test_labels),
    )


def plot_data(X_train, y_train):
    num_row = 2
    num_col = 5  # plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(2 * num_col, 3 * num_row))
    for i in range(10):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(X_train[i * 100].reshape(16, 15), cmap="gray", vmin=0, vmax=6)
        ax.set_title("Label: {}".format(y_train[i * 100]))
    plt.tight_layout()
    plt.savefig("plots/data.png")


def plot_incorrect_images(X_test, y_test, y_pred, model_name):
    wrong_predictions = y_pred != y_test
    num_row = int(np.sqrt(sum(wrong_predictions)))
    num_col = num_row + 1
    fig, axes = plt.subplots(num_row, num_col, figsize=(3 * num_col, 3 * num_row))
    wrong_x, wrong_y, wrong_pred = (
        X_test[wrong_predictions],
        y_test[wrong_predictions],
        np.array(y_pred)[wrong_predictions],
    )
    for i in range(num_row * num_col + num_row):
        try:
            ax = axes[i // num_col, i % num_col]
            ax.imshow(np.array(wrong_x[i]).reshape(16, 15), cmap="gray", vmin=0, vmax=6)
            ax.set_title("Correct: {}. Predicted: {}".format(wrong_y[i], wrong_pred[i]))
        except:
            break
    plt.tight_layout()
    plt.show()
    plt.savefig(f"plots/{model_name}_wrong_predictions.png")


def plot_losses(history, model_name):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(f"Training and Validation loss for {model_name}")
    plt.ylabel("Loss")
    plt.xlabel("No. of Epochs")
    plt.legend(["Train", "Val"], loc="upper right")
    plt.savefig(f"plots/{model_name}_loss.png")


def shift_image(image, dx, dy):
    image = image.reshape((16, 15))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])


def augment_data(X_train, y_train):
    X_train_augmented = [image for image in X_train]
    y_train_augmented = [image for image in y_train]

    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for image, label in zip(X_train, y_train):
            X_train_augmented.append(shift_image(image, dx, dy))
            y_train_augmented.append(label)

    shuffle_idx = np.random.permutation(len(X_train_augmented))
    X_train_augmented = np.array(X_train_augmented)[shuffle_idx]
    y_train_augmented = np.array(y_train_augmented)[shuffle_idx]

    return X_train_augmented, y_train_augmented


def fit_and_predict_model(
    model, model_name, X_train, y_train, X_test, y_test, data_augment
):
    start = time.time()
    if data_augment:
        X_train, y_train = augment_data(X_train, y_train)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    augmentation_sent = "with data augmentation" if data_augment else ""

    plot_incorrect_images(X_test, y_test, y_pred, model_name)
    print(
        f"The model {model_name} {augmentation_sent} took {time.time() - start} seconds to train and results in an accuracy of {accuracy}"
    )


def fit_MLP_model(X_train, y_train, X_test, y_test, data_augment):
    X_train, y_train = shuffle(X_train, y_train, random_state=123)

    if data_augment:
        X_train, y_train = augment_data(X_train, y_train)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["acc"],
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=4
    )
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=2
    )

    callbacks = [early_stopping_callback, reduce_lr_callback]
    start = time.time()

    history = model.fit(
        normalize(X_train),
        tf.keras.utils.to_categorical(y_train),
        validation_split=0.1,
        epochs=50,
        callbacks=callbacks,
    )

    plot_losses(history, "MLP")

    y_pred = model.predict(X_test)
    y_pred = [np.argmax(y) for y in y_pred]

    accuracy = accuracy_score(y_test, y_pred)

    plot_incorrect_images(X_test, y_test, y_pred, "MLP")
    augmentation_sent = "with data augmentation" if data_augment else ""

    print(
        f"The model MLP {augmentation_sent} took {time.time() - start} seconds to train and results in an accuracy of {accuracy}"
    )


def fit_CNN_model(X_train, y_train, X_test, y_test, data_augment):
    X_train, y_train = shuffle(X_train, y_train, random_state=123)

    if data_augment:
        X_train, y_train = augment_data(X_train, y_train)

    model = tf.keras.models.Sequential()

    model.add(
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(16, 15, 1))
    )
    model.add(tf.keras.layers.MaxPool2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.7))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["acc"],
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.001, patience=4
    )
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=2
    )

    callbacks = [early_stopping_callback, reduce_lr_callback]
    start = time.time()

    history = model.fit(
        X_train.reshape(-1, 16, 15, 1),
        tf.keras.utils.to_categorical(y_train),
        validation_split=0.1,
        epochs=50,
        callbacks=callbacks,
    )

    plot_losses(history, "CNN")

    augmentation_sent = "with data augmentation" if data_augment else ""

    y_pred = model.predict(X_test.reshape(-1, 16, 15, 1))
    y_pred = [np.argmax(y) for y in y_pred]

    accuracy = accuracy_score(y_test, y_pred)

    plot_incorrect_images(X_test, y_test, y_pred, "CNN")
    print(
        f"The model CNN {augmentation_sent} took {time.time() - start} seconds to train and results in an accuracy of {accuracy}"
    )
