from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Convert hex strings to integers
def convert_hex_to_int(value):
    if isinstance(value, str):
        return int(value, 16)
    return value

def preprocess_csv(csv_file, shuffle = True):
    # Read the CSV file using pandas
    df = pd.read_csv(csv_file)

    df['Checksum'] = df['Checksum'].apply(convert_hex_to_int)
    df['Options'] = df['Options'].apply(convert_hex_to_int)

    # Convert categorical values to label-encoded integers
    label_encoder = LabelEncoder()
    df['Protocol'] = label_encoder.fit_transform(df['Protocol'])
    df['Flags'] = label_encoder.fit_transform(df['Flags'])

    # Handle NaN values by filling them with a specific value or imputing
    # For example, filling NaN with 0
    df.fillna(0, inplace=True)

    # Remove the "Info" column
    df = df.drop("Info", axis=1)

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

    # print(train_features[:10])
    # print(train_labels[:10])
    # print(test_features[:10])
    # print(test_labels[:10])

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