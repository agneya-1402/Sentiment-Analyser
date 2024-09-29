import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained emotion detection model
model = load_model('SentimentAnalyse/FER_model.h5')  # Assuming you have a pre-trained model

# Load the face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion categories
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define sentiment mapping (positive, negative, neutral)
sentiment_map = {
    'Angry': 'Negative',
    'Disgust': 'Negative',
    'Fear': 'Negative',
    'Happy': 'Positive',
    'Sad': 'Negative',
    'Surprise': 'Positive',
    'Neutral': 'Neutral'
}

# Capture real-time video from webcam
cap = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Preprocess the face region for emotion detection
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))  # Resize to 48x48 (input size for the model)
        face = face / 255.0  # Normalize pixel values
        face = np.reshape(face, (1, 48, 48, 1))  # Reshape for the model input
        
        # Predict the emotion
        emotion_prediction = model.predict(face)
        emotion_label = emotion_labels[np.argmax(emotion_prediction)]
        
        # Map emotion to sentiment
        sentiment = sentiment_map[emotion_label]
        
        # Display the emotion and sentiment on the frame
        cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(frame, f'Sentiment: {sentiment}', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    # Display the frame
    cv2.imshow('Real-time Sentiment Analysis', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
