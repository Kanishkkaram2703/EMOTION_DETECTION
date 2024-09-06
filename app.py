import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

# Load the full model (architecture and weights)
model = load_model(r'H:\KANISHK\projects_null_class\1\full_model.h5')

# Emotion labels
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Streamlit App
st.title("Emotion Detector")
st.write("Upload an image to detect emotions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to detect emotion
def detect_emotion(image):
    # Convert image to numpy array
    img = np.array(image)

    # Ensure the image is in RGB format
    if len(img.shape) == 2:  # Image is already grayscale
        gray_image = img
    elif img.shape[2] == 4:  # Image has an alpha channel (RGBA), remove it
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:  # Regular RGB image
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Load Haar Cascade for face detection
    facec = cv2.CascadeClassifier(r'H:\KANISHK\projects_null_class\1\haarcascade_frontalface_default.xml')
    faces = facec.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=4)

    faces = facec.detectMultiScale(gray_image, 1.3, 5)

    if len(faces) == 0:
        return "No face detected"

    for (x, y, w, h) in faces:
        fc = gray_image[y:y+h, x:x+w]
        
        # Resize to match model's expected input shape (50x50)
        roi = cv2.resize(fc, (50, 50))
        roi = roi[np.newaxis, :, :, np.newaxis]  # Add batch and channel dimension
        
        # Predict the emotion
        pred = EMOTIONS_LIST[np.argmax(model.predict(roi))]

    return pred

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Detect emotion
    st.write("Classifying...")
    emotion = detect_emotion(image)
    st.write(f"Predicted Emotion: {emotion}")
