import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

# Model architecture
from keras import Sequential
from keras.models import load_model
from keras.layers import Dense, GlobalAvgPool2D as GAP, Dropout, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint

# Set the path to the dataset
data_path = 'C:/Users/Admin/Videos/finalYrProject/dataset/raw-img'

# Get a list of class names from the data path
class_names = sorted(os.listdir(data_path))

# Count the number of classes
num_classes = len(class_names)

# Get the number of samples in each class
class_sizes = [len(os.listdir(os.path.join(data_path, name))) for name in class_names]

# Load sampled data
sampled_data_path = r'C:/Users/Admin/Videos/finalYrProject/sample-data'

# Initialize Generator with the specified image transformations and preprocessing
data_generator = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True, 
    vertical_flip=True, 
    rotation_range=20, 
    validation_split=0.2
)

# Load training and validation data
train_data = data_generator.flow_from_directory(
    sampled_data_path, 
    target_size=(256,256), 
    class_mode='binary', 
    batch_size=32, 
    shuffle=True, 
    subset='training'
)

valid_data = data_generator.flow_from_directory(
    sampled_data_path, 
    target_size=(256,256), 
    class_mode='binary', 
    batch_size=32, 
    shuffle=True, 
    subset='validation'
)

# Import the necessary modules
from keras.applications import ResNet152V2

# Define model architecture
name = "ResNet"
base_model = ResNet152V2(include_top=False, input_shape=(256,256,3), weights='imagenet')
base_model.trainable = False

resnet152V2 = Sequential([
    base_model,
    MaxPooling2D(pool_size=(2, 2)),  
    BatchNormalization(),  
    GAP(),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
], name=name)


# Compile the model
resnet152V2.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Define model checkpoint callback
cbs = [ModelCheckpoint(name + ".h5", save_best_only=True)]

# Train the model
history = resnet152V2.fit(
    train_data, 
    validation_data=valid_data,
    epochs=4, 
    callbacks=cbs
)

# Evaluate the model
val_loss, val_accuracy = resnet152V2.evaluate(valid_data, verbose=1)
print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

# Plot training/validation loss and accuracy
def plot_training_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_title('Model Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='upper left')
    
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('Model Loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='upper left')
    
    plt.show()

plot_training_history(history)

# Load the saved model
loaded_model = load_model('ResNet.h5')
loaded_model.summary()
