from numpy import array, ndarray
from numpy.random import shuffle

def load_data_to_trainable_dataset(x: list, y: list) -> tuple[ndarray, ndarray, ndarray, ndarray]:
  """
  This function takes two normal lists of the same length, shuffles and splits them into train and test sets, converts them into numpy arrays and returns them as x_train, x_test, y_train and y_test.

  Parameters:
  x (list): A list of features
  y (list): A list of labels

  Returns:
  x_train (ndarray): A numpy array of the features for training
  x_test (ndarray): A numpy array of the features for testing
  y_train (ndarray): A numpy array of the labels for training
  y_test (ndarray): A numpy array of the labels for testing
  """
  # Assuming x and y are normal lists of the same length
  # Shuffle and split x and y into train and test sets
  data = list(zip(x, y)) # Combine x and y into pairs
  shuffle(data) # Shuffle the data randomly
  train_data = data[:int(0.8 * len(data))] # Take 80% of the data for training
  test_data = data[int(0.8 * len(data)):] # Take 20% of the data for testing

  # Convert the train and test sets into numpy arrays
  x_train, y_train = zip(*train_data) # Unzip the train data into x and y
  x_test, y_test = zip(*test_data) # Unzip the test data into x and y
  x_train = array(x_train) # Convert x_train to numpy array
  y_train = array(y_train) # Convert y_train to numpy array
  x_test = array(x_test) # Convert x_test to numpy array
  y_test = array(y_test) # Convert y_test to numpy array

  # Return the numpy arrays
  return x_train, x_test, y_train, y_test