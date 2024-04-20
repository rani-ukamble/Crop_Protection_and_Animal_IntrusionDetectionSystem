import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from keras.models import load_model


# from tensorflow.keras.models import load_model
from keras.preprocessing import image

model_file_path = 'C:/Users/Admin/Videos/finalYrProject/dataset/animal.h5'

# Load the model
loaded_model = load_model(model_file_path)

# Load the model
# loaded_model = load_model('ResNet152V2.h5')

# Define class mapping
def find_class(f):
    class_mapping = {
        0: "butterfly",
        1: "cat",
        2: "chicken",
        3: "cow",
        4: "dog",
        5: "elephant",
        6: "horse",
        7: "sheep",
        8: "spider",
        9: "squirrel",
        # Add mappings for other classes
    }
    return class_mapping.get(f, "Unknown")

# Load the image using the provided path
img_path = 'C:/Users/Admin/Videos/finalYrProject/dataset/raw-img/mucca/OIP-1b01PBRF9gYAQxQCGK6oKAAAAA.jpeg'
img = image.load_img(img_path, target_size=(256, 256))

# Convert the image to an array and preprocess it
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)  # Add batch dimension
x = x / 255.0  # Normalize the pixel values (assuming pixels are in the range [0, 255])

# Use the loaded model to make predictions
predictions = loaded_model.predict(x)

# Interpret the predictions
predicted_class_index = np.argmax(predictions)
predicted_class_probability = predictions[0][predicted_class_index]

# Load the image
img = mpimg.imread(img_path)

# Display the image
plt.imshow(img)
plt.axis('off')  # Turn off axis

# Display the predicted label
predicted_label = find_class(predicted_class_index)
plt.title("Animal detected: {}".format(predicted_label))

plt.show()

print("Predicted class probability:", predicted_class_probability)
