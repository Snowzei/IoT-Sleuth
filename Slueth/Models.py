from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from numpy import unique

def evaluate_model(model, test_features, test_labels):
    accuracy = model.score(test_features, test_labels)
    print("Test accuracy:", accuracy)

def create_multinomial_naive_bayes(features, labels):
    model = MultinomialNB()
    model.fit(features, labels)
    return model

def create_gaussian_naive_bayes(features, labels):
    model = GaussianNB()
    model.fit(features, labels)
    return model

def create_complement_naive_bayes(features, labels):
    model = ComplementNB()
    model.fit(features, labels)
    return model

def create_svc(features, labels):
    model = SVC()
    model.fit(features, labels)
    return model

def create_knn(features, labels):
    model = KNeighborsClassifier()
    model.fit(features, labels)
    return model

def create_decision_tree(features, labels):
    model = DecisionTreeClassifier()
    model.fit(features, labels)
    return model

def create_random_forest(features, labels):
    model = RandomForestClassifier()
    model.fit(features, labels)
    return model

'''
Neural Networks
----------------------------------------------------------------------------------------------------------------------------------
'''
def evaluate_neural_network(trained_model, test_features, test_labels, normalise=False, encoding=False):
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

def create_neural_network_mk_1(train_features, train_labels, num_epochs=10):
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

def create_neural_network_mk_2(train_features, train_labels, num_epochs=10):
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

def create_neural_network_mk_3(train_features, train_labels, num_epochs=10):
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

def create_neural_network_mk_4(train_features, train_labels, num_epochs=10):
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