# 📚 Deep Learning Classifiers: Cats vs Dogs & SMS Spam Detection

This project showcases two deep learning models implemented in a single Jupyter Notebook:

1. Cat and Dog Image Classifier using a Convolutional Neural Network (CNN).

2. SMS Text Classifier using a Bidirectional Long Short-Term Memory (Bi-LSTM) Recurrent Neural Network (RNN).

Both models are built using TensorFlow 2.x and Keras.


## 🐱🐶 Cat and Dog Image Classifier

### 🔍 Overview
A binary image classifier trained to distinguish between images of cats and dogs using a Convolutional Neural Network (CNN).

### 🛠️ Data Preparation
- ImageDataGenerator for image augmentation and normalization.

- Directory Structure:
```bash
├── train/
│   ├── cats/
│   └── dogs/
├── validation/
│   ├── cats/
│   └── dogs/
└── test/
    ├── unknown/
```
- flow_from_directory() was used to load images and auto-label them.

### 🧠 Model Architecture
- Conv2D + MaxPooling2D layers to extract spatial features.
- Dropout for regularization.
- Dense layers with a sigmoid output for binary classification.

### 🏁 Training & Evaluation
- Used early stopping to prevent overfitting.
- Visualized accuracy and loss during training.
- Model tested on unseen test images.


## 💬 SMS Text Classifier

### 🔍 Overview
A binary text classifier trained to classify SMS messages as ham (not spam) or spam using a Bidirectional LSTM model.

### 🛠️ Data Preparation
- SMS Spam Collection Dataset loaded as a DataFrame.
- Labels encoded (ham → 0, spam → 1).
- Keras TextVectorization used to tokenize and vectorize SMS messages.

### 🧠 Model Architecture
- Embedding layer to learn word embeddings.
- Bidirectional LSTM for understanding context from both directions.
- Dense layers with sigmoid activation for binary classification.

### 🏁 Training & Evaluation
- Used early stopping based on validation loss.
- Custom predict_message() function for real-time predictions.


## 📊 Summary Comparison
| Project               | Input Type      | Model         | Output       |
|-----------------------|-----------------|---------------|--------------|
| Cat & Dog Classifier  | JPG images      | CNN           | “cat”/“dog”  |
| SMS Text Classifier   | Plain text SMS  | Bi-LSTM RNN   | “ham”/“spam” |

## 🧰 Requirements
```bash
pip install tensorflow pandas numpy matplotlib
```

## 📁 File Structure
```bash
📦 neural_networks/
├── sms_t_c.ipynb    # The main Jupyter notebook
└── README.md        # Project description
```

## 📌 Note
All code is implemented inside one Jupyter notebook. You can run each section independently to train and test the models.

## 🏷️ Project Origin
This project is a solution to a [freeCodeCamp](https://www.freecodecamp.org/) deep learning project challenge.

## 📎 License
This project is licensed under the [MIT License](https://mit-license.org/).