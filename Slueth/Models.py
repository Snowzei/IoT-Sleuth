import tensorflow as tf
import numpy as np

def train_model(model: tf.keras.Model, x_train: np.ndarray, y_train: np.ndarray,
                x_val: np.ndarray, y_val: np.ndarray, epochs: int) -> tf.keras.callbacks.History:
    """Trains a model on the given data.

    Args:
        model (tf.keras.Model): A TensorFlow model object.
        x_train (np.ndarray): The training data array.
        y_train (np.ndarray): The training labels array.
        x_val (np.ndarray): The validation data array.
        y_val (np.ndarray): The validation labels array.
        epochs (int): The number of epochs to train.

    Returns:
        tf.keras.callbacks.History: A history object that contains the loss and accuracy values for each epoch.
    """
    # Fit the model to the training data
    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        epochs=epochs)

    # Return the history object
    return history

'''
-------------------------------
Classification Models:
- naive bayes
- Support vector machine
- K Nearest Neighbour
- decision tree/ Random Forest
-------------------------------
'''

def create_naive_bayes_model() -> tf.keras.Model:
    """Creates a naive bayes model for predicting device types.

    Returns:
        tf.keras.Model: A TensorFlow model object.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam())
    
    return model

def create_SVM_model() -> tf.keras.Model:
    """Creates a Support vector machine model for predicting device types.

    Returns:
        tf.keras.Model: A TensorFlow model object.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_classes, activation='linear')
    ])
    model.compile(loss=tf.keras.losses.CategoricalHinge(),
                  optimizer=tf.keras.optimizers.SGD())
    
    return model

def create_KNN_model() -> tf.keras.Model:
    """Creates a K Nearest Neighbour model for predicting device types.

    Returns:
        tf.keras.Model: A TensorFlow model object.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam())
    
    return model

def create_random_forest_model() -> tf.keras.Model:
    """Creates a Random Forest model for predicting device types.

    Returns:
        tf.keras.Model: A TensorFlow model object.
    """
    model = tf.keras.experimental.RandomForest(tf.keras.Input(shape=(None,)),
                                               num_trees=10,
                                               max_depth=5,
                                               num_classes=num_classes)

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam())
    
    return model

'''
-------------------------------
Regression Models:
- Multi-Layer Perceptron
-------------------------------
'''

def create_MLP_model() -> tf.keras.Model:
    """Creates a multi-layer perceptron model for predicting device types.

    Returns:
        tf.keras.Model: A TensorFlow model object.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam())
    
    return model