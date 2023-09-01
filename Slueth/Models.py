# Import sklearn classes/functions
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Import tensorflow classes/functions
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Import numpy features
from numpy import unique, ndarray

'''
Classifiers
----------------------------------------------------------------------------------------------------------------------------------
'''

def evaluate_model(model: any, test_features: ndarray, test_labels:ndarray, scaling_and_processing:bool=False):
    '''
    Evaluates a model.

    Parameters:
        model (any): The trained model from the sklearn library. 
        test_features (ndarray): Input features as a 2D NumPy array.
        test_labels (ndarray): Target labels as a 1D NumPy array.
        scaling_and_processing (bool, optional): Whether to apply scaling and processing to test features.
    '''
    if scaling_and_processing:
        # Preprocess the features
        scaler = StandardScaler()
        imputer = SimpleImputer(strategy='mean')
        # Scale and impute the features
        scaled_features = scaler.fit_transform(test_features)
        preprocessed_features = imputer.fit_transform(scaled_features)
        accuracy = model.score(preprocessed_features, test_labels)
    else:
        accuracy = model.score(test_features, test_labels)
    print("Test accuracy:", accuracy)

def create_multinomial_naive_bayes(features: ndarray, labels: ndarray) -> MultinomialNB:
    '''
    Creates and trains a Multinomial Naive Bayes model.

    Parameters:
        features (ndarray): Input features as a 2D NumPy array.
        labels (ndarray): Target labels as a 1D NumPy array.

    Returns:
        MultinomialNB: Trained Multinomial Naive Bayes model.
    '''
    model = MultinomialNB()
    model.fit(features, labels)
    return model

def create_gaussian_naive_bayes(features: ndarray, labels: ndarray) -> GaussianNB:
    '''
    Creates and trains a Gaussian Naive Bayes model.

    Parameters:
        features (ndarray): Input features as a 2D NumPy array.
        labels (ndarray): Target labels as a 1D NumPy array.

    Returns:
        GaussianNB: Trained Gaussian Naive Bayes model.
    '''
    model = GaussianNB()
    model.fit(features, labels)
    return model

def create_complement_naive_bayes(features: ndarray, labels: ndarray) -> ComplementNB:
    '''
    Creates and trains a Complement Naive Bayes model.

    Parameters:
        features (ndarray): Input features as a 2D NumPy array.
        labels (ndarray): Target labels as a 1D NumPy array.

    Returns:
        ComplementNB: Trained Complement Naive Bayes model.
    '''
    model = ComplementNB()
    model.fit(features, labels)
    return model

def create_svc(features: ndarray, labels: ndarray) -> SVC:
    '''
    Creates and trains a C-Support Vector Classification model.

    Parameters:
        features (ndarray): Input features as a 2D NumPy array.
        labels (ndarray): Target labels as a 1D NumPy array.

    Returns:
        SVC: Trained C-Support Vector Classification model.
    '''
    model = SVC()
    model.fit(features, labels)
    return model

def create_knn(features: ndarray, labels: ndarray) -> KNeighborsClassifier:
    '''
    Creates and trains a K-Nearest Neighbors model.

    Parameters:
        features (ndarray): Input features as a 2D NumPy array.
        labels (ndarray): Target labels as a 1D NumPy array.

    Returns:
        KNeighborsClassifier: Trained K-Nearest Neighbors model.
    '''
    model = KNeighborsClassifier()
    model.fit(features, labels)
    return model

def create_decision_tree(features: ndarray, labels: ndarray) -> DecisionTreeClassifier:
    '''
    Creates and trains a Decision Tree Classifier.

    Parameters:
        features (ndarray): Input features as a 2D NumPy array.
        labels (ndarray): Target labels as a 1D NumPy array.

    Returns:
        DecisionTreeClassifier: Trained Decision Tree classifier model.
    '''
    # Preprocess the features
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='mean')
    # Scale and impute the features
    scaled_features = scaler.fit_transform(features)
    preprocessed_features = imputer.fit_transform(scaled_features)
    # Create and train the model
    model = DecisionTreeClassifier()
    model.fit(preprocessed_features, labels)
    return model

def create_random_forest(features: ndarray, labels: ndarray) -> RandomForestClassifier:
    '''
    Creates and trains a Random Forest classifier.

    Parameters:
        features (ndarray): Input features as a 2D NumPy array.
        labels (ndarray): Target labels as a 1D NumPy array.

    Returns:
        RandomForestClassifier: Trained Random Forest classifier model.
    '''
    # Preprocess the features
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='mean')
    # Scale and impute the features
    scaled_features = scaler.fit_transform(features)
    preprocessed_features = imputer.fit_transform(scaled_features)
    # Create and train the model
    model = RandomForestClassifier()
    model.fit(preprocessed_features, labels)
    return model

'''
Neural Networks
----------------------------------------------------------------------------------------------------------------------------------
'''

def evaluate_neural_network(trained_model:Sequential, test_features:ndarray, test_labels:ndarray, normalise:bool=False, encoding:bool=False):
    '''
    Evaluates a neural network using the test dataset.

    Parameters:
        trained_model (Sequential): The trained model.
        test_features (ndarray): Testing input features as a 2D NumPy array.
        test_labels (ndarray): Testing target labels as a 1D NumPy array.
        normalise (bool, optional): Whether to apply encoding to test features. 
        encoding (bool, optional): Whether to apply normalising to test labels.
    '''    
    if normalise:
        scaler = StandardScaler()
        test_features = scaler.fit_transform(test_features)
    if encoding:
        label_encoder = LabelEncoder()
        test_labels = label_encoder.fit_transform(test_labels)
    loss, accuracy = trained_model.evaluate(test_features, test_labels)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
    return

def create_neural_network_mk_1(train_features:ndarray, train_labels:ndarray, num_epochs:int=10) -> Sequential:
    '''
    Creates and trains a neural network using the tensorflow library. This is version 1 of the model.

    Parameters:
        features (ndarray): Input features as a 2D NumPy array.
        labels (ndarray): Target labels as a 1D NumPy array.

    Returns:
        Sequential: Trained neural network model.
    '''
    # Get the num_classes from the data
    num_classes = len(unique(train_labels))
    # Create a neural network model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(train_features.shape[1],)),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(train_features, train_labels, epochs=num_epochs, verbose=1)
    return model

def create_neural_network_mk_2(train_features:ndarray, train_labels:ndarray, num_epochs:int=10) -> Sequential:
    '''
    Creates and trains a neural network using the tensorflow library. This is version 2 of the model.

    Parameters:
        features (ndarray): Input features as a 2D NumPy array.
        labels (ndarray): Target labels as a 1D NumPy array.

    Returns:
        Sequential: Trained neural network model.
    '''
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(train_labels)
    # Get the num_classes from the data
    num_classes = len(label_encoder.classes_)
    # Create a neural network model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(train_features.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(train_features, encoded_labels, epochs=num_epochs, verbose=1)
    return model

def create_neural_network_mk_3(train_features:ndarray, train_labels:ndarray, num_epochs:int=10) -> Sequential:
    '''
    Creates and trains a neural network using the tensorflow library. This is version 3 of the model.

    Parameters:
        features (ndarray): Input features as a 2D NumPy array.
        labels (ndarray): Target labels as a 1D NumPy array.

    Returns:
        Sequential: Trained neural network model.
    '''
    # Normalise features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(train_features)
    # Get the num_classes from the data
    num_classes = len(unique(train_labels))
    # Create a neural network model
    model = Sequential([
        Dense(256, activation='relu', input_shape=(scaled_features.shape[1],)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(scaled_features, train_labels, epochs=num_epochs, verbose=1)
    return model

def create_neural_network_mk_4(train_features:ndarray, train_labels:ndarray, num_epochs:int=10) -> Sequential:
    '''
    Creates and trains a neural network using the tensorflow library. This is version 4 of the model.

    Parameters:
        features (ndarray): Input features as a 2D NumPy array.
        labels (ndarray): Target labels as a 1D NumPy array.

    Returns:
        Sequential: Trained neural network model.
    '''
    # Normalise features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(train_features)
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(train_labels)
    # Get the num_classes from the data
    num_classes = len(label_encoder.classes_)
    # Create a neural network model
    model = Sequential([
        Dense(256, activation='relu', input_shape=(scaled_features.shape[1],)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='elu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='selu'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(scaled_features, encoded_labels, epochs=num_epochs, verbose=1)
    return model