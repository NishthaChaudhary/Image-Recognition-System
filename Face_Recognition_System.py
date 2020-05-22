#!/usr/bin/env python
# coding: utf-8

# # Face Recognition System

# ### Have you ever wondered how Facebook identifies who's in your photos without you even having tag anyone?? This is Face Recognition Technique. First, we will walk through each step of face recognition process. Then, we will build a cutting edge face recognition system that you can reuse in your own projects. Finally, we will see how face recognition can be applied to a variety of situations and projects

# ## FIND FACES IN PICTURE

# In[2]:


#pip install dlib==19.8.1
#pip install face_recognition==1.2.2
#pip install pillow
#pip install numpy
#pip install pip>=9.0.0
#pip install pandas
#pip install matplotlib
#pip install Keras==2.1.6
#pip install h5py
#pip install pillow
#pip install scikit-learn
#pip install scipy
#pip install tensorboard==1.8.0
#pip install google-api-python-client==1.6.7
#pip install joblib


# In[71]:


#import win32api
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import face_recognition
from IPython.display import display
from PIL import Image, ImageDraw
import numpy as np


# In[4]:


import tensorflow as tf
import keras
from keras.preprocessing import image


# In[6]:


# Load the jpg file into a numpy array
im=image.load_img("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/unknown_2.jpg")
#image = face_recognition.load_image_file("C:/Users/nisht/Desktop/MITA/LinkedIn_Learning/Ex_Files_Deep_Learning_Face_Recog/Exercise Files/Windows Exercise Files/Ch07/formal_pic.jpg")
display(im)


# ### Lets see on their indiviadual pictures:

# In[12]:


#First learning pic--ALICE
p_1 = Image.open("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/person_1.jpg")
display(p_1)


# In[13]:


#Second learning pic-- MAY
p_2 = Image.open("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/person_2.jpg")
display(p_2)


# In[14]:


#Third learning pic-- BRIAN
p_3 = Image.open("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/person_3.jpg")
display(p_3)


# ### This is a system of running face recognition on a single image and drawing a box around each person that was identified.

# In[64]:


#Load a sample pictures and learn how to recognize it
person_1 = face_recognition.load_image_file("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/person_1.jpg")
person_1_encoding = face_recognition.face_encodings(person_1)[0]

person_2 = face_recognition.load_image_file("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/person_2.jpg")
person_2_encoding = face_recognition.face_encodings(person_2)[0]

person_3 = face_recognition.load_image_file("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/person_3.jpg")
person_3_encoding = face_recognition.face_encodings(person_3)[0]


# In[65]:


# Create arrays of known face encodings and their names
known_face_encodings = [
    person_1_encoding,
    person_2_encoding,
    person_3_encoding
    
]
known_face_names = [
    "Alice",
    "May",
    "Brian"
]
print('Learned encoding for', len(known_face_encodings), 'images.')


# ### Finally, we load the image we looked at in the first cell, find the faces in the image and compare them with the encodings the library generated in the previous step. We can see that library now correctly recognizes Alice, May and Brian in the input.

# In[76]:


# Load an image with an unknown face
from PIL import ImageFont

unknown_image = face_recognition.load_image_file("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/unknown_2.jpg")

fontsize = 15
font = ImageFont.truetype("arial.ttf", fontsize)

# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
# See http://pillow.readthedocs.io/ for more about PIL/Pillow
pil_image = Image.fromarray(unknown_image)

# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 20, bottom - text_height - 5), name, fill=(255, 255, 255, 255),font=font)


# Remove the drawing library from memory as per the Pillow docs
del draw

# Display the resulting image
display(pil_image)


# ## MAKE UP APP

# In[44]:


# Load an image with an unknown face
unknown_image = face_recognition.load_image_file("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/unknown_2.jpg")


# In[49]:


number_of_faces = len(face_landmarks_list)
print("I found {} face(s) in this photograph.".format(number_of_faces))


# In[50]:


# Load the image into a Python Image Library object so that we can draw on top of it and display it
pil_image = PIL.Image.fromarray(unknown_image)

# Create a PIL drawing object to be able to draw lines later
draw = PIL.ImageDraw.Draw(pil_image)

# Loop over each face
for face_landmarks in face_landmarks_list:

    # Loop over each facial feature (eye, nose, mouth, lips, etc)
    for name, list_of_points in face_landmarks.items():

        # Print the location of each facial feature in this image
        print("The {} in this face has the following points: {}".format(name, list_of_points))

        # Let's trace out each facial feature in the image with a line!
        draw.line(list_of_points, fill="red", width=2)

pil_image


# In[48]:


# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(unknown_image)

pil_image = Image.fromarray(unknown_image)
for face_landmarks in face_landmarks_list:
    d = ImageDraw.Draw(pil_image, 'RGBA')

    # Make the eyebrows into a nightmare
    d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
    d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
    d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
    d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

    # Gloss the lips
    d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
    d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
    d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
    d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

    # Sparkle the eyes
    d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
    d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

    # Apply some eyeliner
    d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
    d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

display(pil_image)


# ### FIND THE UNKNOWN IMAGE

# In[79]:


import face_recognition

# Load the known images
image_of_person_1 = face_recognition.load_image_file("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/nishtha.jpg")
image_of_person_2 = face_recognition.load_image_file("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/person_2.jpg")
image_of_person_3 = face_recognition.load_image_file("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/person_3.jpg")

# Get the face encoding of each person. This can fail if no one is found in the photo.
person_1_face_encoding = face_recognition.face_encodings(image_of_person_1)[0]
person_2_face_encoding = face_recognition.face_encodings(image_of_person_2)[0]
person_3_face_encoding = face_recognition.face_encodings(image_of_person_3)[0]

# Create a list of all known face encodings
known_face_encodings = [
    person_1_face_encoding,
    person_2_face_encoding,
    person_3_face_encoding
]

# Load the image we want to check
unknown_image = face_recognition.load_image_file("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/unknown_2.jpg")

# Get face encodings for any people in the picture
unknown_face_encodings = face_recognition.face_encodings(unknown_image)

# There might be more than one person in the photo, so we need to loop over each face we found
for unknown_face_encoding in unknown_face_encodings:

    # Test if this unknown face encoding matches any of the three people we know
    results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding, tolerance=0.6)

    name = "Unknown"

    if results[0]:
        name = "Person 1"
    elif results[1]:
        name = "Person 2"
    elif results[2]:
        name = "Person 3"
    print(f"Found {name} in the photo!")


# ### SIMILAR LOOKING IPERSON

# #### Pass a person's image over a set of images to find the similar looking person.

# In[58]:


#Display the known image

#from pathlib import Path
#from PIL import Image

known= Image.open("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/formal_pic.jpg")
# Load the image of the person we want to find similar people for
known_image = face_recognition.load_image_file("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/formal_pic.jpg")
display(known)


# In[59]:


#Look into a set of images to find a similar image

# Encode the known image
known_image_encoding = face_recognition.face_encodings(known_image)[0]

# Variables to keep track of the most similar face match we've found
best_face_distance = 1.0
best_face_image = None

# Loop over all the images we want to check for similar people
for image_path in Path("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch07/people").glob("*.png"):
    # Load an image to check
    unknown_image = face_recognition.load_image_file(image_path)

    # Get the location of faces and face encodings for the current image
    face_encodings = face_recognition.face_encodings(unknown_image)

    # Get the face distance between the known person and all the faces in this image
    face_distance = face_recognition.face_distance(face_encodings, known_image_encoding)[0]

    # If this face is more similar to our known image than we've seen so far, save it
    if face_distance < best_face_distance:
        # Save the new best face distance
        best_face_distance = face_distance
        # Extract a copy of the actual face image itself so we can display it
        best_face_image = unknown_image

# Display the face image that we found to be the best match!
pil_image = Image.fromarray(best_face_image)
pil_image


# #### OHH YES!! This is me :)
