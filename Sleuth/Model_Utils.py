from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Convert hex strings to integers
def convert_hex_to_int(value:str)->int:
    '''
    Convert a hexadecimal string to an integer.

    Args:
        value (str): The hexadecimal string to be converted.

    Returns:
        int: The integer representation of the hexadecimal value.

    This function takes a hexadecimal string as input and returns the corresponding
    integer representation. If the input is not a valid hexadecimal string, it
    returns the input value as is. Use this function to convert hexadecimal values
    to integers for further processing.
    '''
    if isinstance(value, str):
        return int(value, 16)
    return value

def preprocess_csv(csv_file:str, shuffle:bool = True, drop_length:bool = False, drop_window_size:bool = False)->tuple:
    '''
    Preprocesses a CSV file containing IoT network data for machine learning.

    Args:
        csv_file (str): The path to the CSV file containing IoT network data.
        shuffle (bool, optional): Whether to shuffle the dataset. Default is True.

    Returns:
        tuple: A tuple containing the following NumPy arrays:
            - train_features (ndarray): Features for training data.
            - train_labels (ndarray): Labels for training data.
            - test_features (ndarray): Features for testing data.
            - test_labels (ndarray): Labels for testing data.

    This function reads the CSV file, performs data preprocessing tasks including
    converting hex values, label encoding, handling NaN values, and optionally shuffling
    the dataset. It then splits the data into training and testing sets, returning
    NumPy arrays for further machine learning tasks.
    '''
    # Read the CSV file using pandas
    df = pd.read_csv(csv_file)
    df['Checksum'] = df['Checksum'].apply(convert_hex_to_int)
    df['Options'] = df['Options'].apply(convert_hex_to_int)
    # Convert categorical values to label-encoded integers
    label_encoder = LabelEncoder()
    df['Protocol'] = label_encoder.fit_transform(df['Protocol'])
    df['Flags'] = label_encoder.fit_transform(df['Flags'])
    # Handle NaN values by filling them with a specific value
    df.fillna(0, inplace=True)
    # Remove the "Info" column
    df = df.drop("Info", axis=1)
    if drop_length:
        df = df.drop("Length", axis=1)
    if drop_window_size:
        df = df.drop("Window Size", axis=1)
    # Shuffle the dataset
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    # Split features and labels
    features = df.drop('Label', axis=1)
    labels = df['Label']
    # Convert features and labels to NumPy arrays with appropriate data types
    features_array = features.to_numpy().astype(float)  # Convert to float to handle mixed types
    labels_array = labels.to_numpy()
    #split the features and labels further into training and testing data
    split_index = int(len(features_array) * 0.8)  # Index to split at
    train_features = features_array[:split_index]  # First 80% of the array
    test_features = features_array[split_index:] # Last 20% of the array
    train_labels = labels_array[:split_index]  # First 80% of the array
    test_labels = labels_array[split_index:] # Last 20% of the array
    # Convert the NumPy arrays to TensorFlow tensors and create the dataset
    # dataset = tf.data.Dataset.from_tensor_slices((features_array, labels_array))

    return train_features, train_labels, test_features, test_labels

'''
def dataset_into_features_and_labels(dataset):
    # Assuming dataset is a TensorFlow dataset with features and labels

    # Split the TensorFlow dataset into features and labels
    features_dataset = dataset.map(lambda x, y: x)
    labels_dataset = dataset.map(lambda x, y: y)

    # Convert features and labels datasets into arrays
    features_array = np.array(list(features_dataset.as_numpy_iterator()))
    labels_array = np.array(list(labels_dataset.as_numpy_iterator()))
    return features_array, labels_array

-----------------------
def split_dataset(dataset):
    # Split the dataset into train, validation, and test sets
    train_size = int(0.8 * len(dataset))

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    return train_dataset, test_dataset
'''