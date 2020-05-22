# Image recognition using pre-trained model

from keras.applications import vgg16
# Load Keras' VGG16 model that was pre-trained against the ImageNet database
model = vgg16.VGG16()


# Sample IDea

#Let's try to recognize for one image
# Load the image file, resizing it to 224x224 pixels (required by this model)
img = image.load_img("C:/Users/nisht/Anaconda3_n/envs/LL/Image_Recog/Ch05/bay.jpg", target_size=(224, 224))
img

# Convert the image to a numpy array
x = image.img_to_array(img)

# Add a fourth dimension (since Keras expects a list of images)
x = np.expand_dims(x, axis=0)

# Normalize the input image's pixel values to the range used when training the neural network
x = vgg16.preprocess_input(x)

# Run the image through the deep neural network to make a prediction
predictions = model.predict(x)

# Look up the names of the predicted classes. Index zero is the results for the first image.
predicted_classes = vgg16.decode_predictions(predictions)

print("Top predictions for this image:")

for imagenet_id, name, likelihood in predicted_classes[0]:
    print("Prediction: {} - {:2f}".format(name, likelihood))


# Feature Extraction

#from pathlib import Path
#import numpy as np
#import joblib
#from keras.preprocessing import image
#from keras.applications import vgg16

# Path to folders with training data
dog_path = Path("C:/Users/nisht/Anaconda3_n/envs/LL/Image_Recog/Ch05/training_data") / "dogs"
not_dog_path = Path("C:/Users/nisht/Anaconda3_n/envs/LL/Image_Recog/Ch05/training_data") / "not_dogs"

images = []
labels = []

# Load all the not-dog images
for img in not_dog_path.glob("*.png"):
    # Load the image from disk
    img = image.load_img(img)

    # Convert the image to a numpy array
    image_array = image.img_to_array(img)

    # Add the image to the list of images
    images.append(image_array)

    # For each 'not dog' image, the expected value should be 0
    labels.append(0)

# Load all the dog images
for img in dog_path.glob("*.png"):
    # Load the image from disk
    img = image.load_img(img)

    # Convert the image to a numpy array
    image_array = image.img_to_array(img)

    # Add the image to the list of images
    images.append(image_array)

    # For each 'dog' image, the expected value should be 1
    labels.append(1)

# Create a single numpy array with all the images we loaded
x_train = np.array(images)

# Also convert the labels to a numpy array
y_train = np.array(labels)

# Normalize image data to 0-to-1 range
x_train = vgg16.preprocess_input(x_train)

# Load a pre-trained neural network to use as a feature extractor
pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Extract features for each image (all in one pass)
features_x = pretrained_nn.predict(x_train)

# Save the array of extracted features to a file
joblib.dump(features_x, "C:/Users/nisht/Anaconda3_n/envs/LL/Image_Recog/Ch05/x_train.dat")

# Save the matching array of expected values to a file
joblib.dump(y_train, "C:/Users/nisht/Anaconda3_n/envs/LL/Image_Recog/Ch05/y_train.dat")


# TRAINING WITH EXTRACTED FEATURES

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

# Load data set
x_train = joblib.load("C:/Users/nisht/Anaconda3_n/envs/LL/Image_Recog/Ch05/x_train.dat")
y_train = joblib.load("C:/Users/nisht/Anaconda3_n/envs/LL/Image_Recog/Ch05/y_train.dat")

# Create a model and add layers
model = Sequential()

model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=['accuracy']
)

# Train the model
model.fit(
    x_train,
    y_train,
    epochs=10,
    shuffle=True
)

# Save neural network structure
model_structure = model.to_json()
f = Path("C:/Users/nisht/Anaconda3_n/envs/LL/Image_Recog/Ch05/model_structure.json")
f.write_text(model_structure)

# Save neural network's trained weights
model.save_weights("C:/Users/nisht/Anaconda3_n/envs/LL/Image_Recog/Ch05/model_weights.h5")


# MAKING PREDICTIONS

from keras.models import model_from_json
#from pathlib import Path
#from keras.preprocessing import image
#import numpy as np
#from keras.applications import vgg16

# Load the json file that contains the model's structure
f = Path("C:/Users/nisht/Anaconda3_n/envs/LL/Image_Recog/Ch05/model_structure.json")
model_structure = f.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights("C:/Users/nisht/Anaconda3_n/envs/LL/Image_Recog/Ch05/model_weights.h5")

# Load an image file to test, resizing it to 64x64 pixels (as required by this model)
img = image.load_img("C:/Users/nisht/Anaconda3_n/envs/LL/Image_Recog/Ch05/not_dog.png", target_size=(64, 64))

# Convert the image to a numpy array
image_array = image.img_to_array(img)

# Add a forth dimension to the image (since Keras expects a bunch of images, not a single image)
images = np.expand_dims(image_array, axis=0)

# Normalize the data
images = vgg16.preprocess_input(images)

# Use the pre-trained neural network to extract features from our test image (the same way we did to train the model)
feature_extraction_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
features = feature_extraction_model.predict(images)

# Given the extracted features, make a final prediction using our own model
results = model.predict(features)

# Since we are only testing one image with possible class, we only need to check the first result's first element
single_result = results[0][0]

# Print the result
print("Likelihood that this image contains a dog: {}%".format(int(single_result * 100)))
