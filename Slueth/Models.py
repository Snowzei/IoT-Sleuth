import tensorflow as tf
import numpy as np
from tensorflow_probability.python.distributions import Dirichlet, Beta

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
- Naive bayes
- Support vector machine
- K Nearest Neighbour
- Decision tree/ Random Forest
-------------------------------
'''

def create_naive_bayes_model(num_classes: int) -> tf.keras.Model:
    """Creates a naive bayes model for predicting device types.

    Args:
        int: num_classes: The number of different devices, this is a parameter for ease of use later.

    Returns:
        tf.keras.Model: A TensorFlow model object.
    """
    # define the prior distribution for the class probabilities
    pi = Dirichlet(concentration=tf.ones(num_classes))

    # define the prior distribution for the feature probabilities
    theta = Beta(concentration0=tf.ones([num_classes, 10]), concentration1=tf.ones([num_classes, 10]))

    # define the posterior distribution for the class and feature probabilities using variational inference
    q_pi = Dirichlet(concentration=tf.Variable(tf.ones(num_classes)))
    q_theta = Beta(concentration0=tf.Variable(tf.ones([num_classes, 10])), concentration1=tf.Variable(tf.ones([num_classes, 10])))

    # create a simple naive bayes model with TensorFlow
    model = tf.keras.Model(inputs=[], outputs=[q_pi, q_theta])
    
    return model

def create_SVM_model(num_classes: int) -> tf.keras.Model:
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

def create_KNN_model(num_classes: int) -> tf.keras.Model:
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

def create_random_forest_model(num_classes: int) -> tf.keras.Model:
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