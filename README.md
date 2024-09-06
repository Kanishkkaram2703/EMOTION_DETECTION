# EMOTION_DETECTION
An simple emotion detector using ML


Emotion Detection Using Deep Learning
This project is designed to detect emotions in human faces using a Convolutional Neural Network (CNN) model. The solution is built using TensorFlow/Keras, and Streamlit is used to create an interactive web interface where users can upload an image and get the predicted emotion. The project includes a pre-trained model that identifies various emotions from facial expressions. Below is a detailed explanation of how the emotion detection system works.

Project Structure
model.json: The architecture of the pre-trained model in JSON format.
full_model.h5: The model's pre-trained weights.
haarcascade_frontalface_default.xml: The pre-trained Haar Cascade classifier for face detection.
app.py: The main Streamlit application file.
README.md: Documentation explaining the project.
Key Features
Emotion Detection: The model can detect seven emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
Face Detection: Uses OpenCV’s Haar Cascade classifier to detect faces within the uploaded image before passing it to the emotion detection model.
Streamlit Interface: Provides a simple and user-friendly interface for users to upload images and view results.
Installation and Setup
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/emotion-detector.git
cd emotion-detector
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the application:

bash
Copy code
streamlit run app.py
Code Explanation
1. Loading the Pre-trained Model
python
Copy code
from tensorflow.keras.models import load_model

model = load_model('full_model.h5')
We load the CNN model trained for emotion detection. This model is built using a set of images with labeled emotions, and it has been trained to predict the emotion of the person in the image based on their facial expression.

2. Face Detection
python
Copy code
import cv2
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = facec.detectMultiScale(gray_image, 1.3, 5)
We use OpenCV's Haar Cascade to detect faces in the uploaded image. The detectMultiScale() method scans the grayscale image and returns a list of coordinates representing faces.

If no faces are found, the model returns "No face detected."
3. Preprocessing the Detected Face
python
Copy code
roi = cv2.resize(fc, (48, 48))
roi = roi[np.newaxis, :, :, np.newaxis]
Once a face is detected, it is cropped and resized to a 48x48 pixel image, which is the input size required by the pre-trained CNN model. The model expects a batch of images, so we expand the dimensions using np.newaxis.

4. Emotion Prediction
python
Copy code
pred = EMOTIONS_LIST[np.argmax(model.predict(roi))]
The preprocessed image is passed to the model, which returns a prediction vector. The vector contains probabilities for each of the seven possible emotions. The function np.argmax() retrieves the index of the highest probability, which corresponds to the predicted emotion.

5. Displaying Results
python
Copy code
st.image(image, caption='Uploaded Image.', use_column_width=True)
st.write(f"Predicted Emotion: {emotion}")
Streamlit is used to display the uploaded image along with the predicted emotion. The user interface is interactive, allowing users to easily upload new images and view their results.

Conclusion
This project showcases how a combination of deep learning models and traditional computer vision techniques can be used to create an effective emotion detection system. It can be expanded further with real-time video processing, a larger set of emotions, or even integration into web applications for user feedback analysis.
