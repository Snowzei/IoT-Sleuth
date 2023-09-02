# IoT Device Identification with Machine Learning - IoT Slueth

![Windows](https://img.shields.io/badge/Platform-Windows-blue?logo=windows)
![Linux](https://img.shields.io/badge/Platform-Linux-green?logo=linux)
![Language](https://img.shields.io/badge/Language-Python-brightgreen)
![Notebook](https://img.shields.io/badge/Environment-Jupyter%20Notebook-orange)
[![Library](https://img.shields.io/badge/Library-TensorFlow%20%7C%20scikit--learn-blueviolet)](https://www.tensorflow.org/)
[![Library](https://img.shields.io/badge/Library-scikit--learn-blueviolet)](https://scikit-learn.org/stable/)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

Welcome to the IoT Device Identification project IoT-Slueth! This project explores various machine learning models, including classic identifiers like Multinomial Naive Bayes, Random Forest, Decision Tree, K-Nearest Neighbors, and C-Support Vector Classification, alongside a custom neural network model trained on benign traffic. The goal is to accurately identify and classify IoT devices within a network.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Models Explored](#models-explored)
- [Dataset](#dataset)
- [License](#license)

## Introduction

The Internet of Things (IoT) has become an integral part of our daily lives, encompassing a wide range of devices. Ensuring the security and efficient management of these devices is crucial. This project focuses on the identification of IoT devices within a network using machine learning techniques.

## Installation

To set up the project and install the required libraries, follow these steps:

1. Clone the repository:

```
git clone https://github.com/yourusername/iot-device-identification.git
```

2. Navigate to the project directory:

```
cd iot-device-identification
```

3. Create a virtual environment (optional but recommended):

```
python -m venv venv
```


4. Activate the virtual environment:
- On Windows:
  ```
  venv\Scripts\activate
  ```
- On macOS and Linux:
  ```
  source venv/bin/activate
  ```

5. Install the required libraries from the `requirements.txt` file:

```
pip install -r requirements.txt
```

## Usage

To use the project and explore different machine learning models and the custom neural network, follow the Jupyter Notebook or Python script examples provided in the project's directory.

## Models Explored

This project explores several machine learning models for IoT device identification:

- Multinomial Naive Bayes
- Random Forest
- Decision Tree
- K-Nearest Neighbors
- C-Support Vector Classification (SVM)
- Custom Neural Network

Each model is designed to classify IoT devices based on network traffic patterns and behavior.

## Dataset

The project uses a dataset containing network traffic data for training and testing. The dataset includes features such as Length, Protocol, Time To Live, Window Size, Options, Flags, Checksum and Packet Info. It is used to train and evaluate the machine learning models.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.