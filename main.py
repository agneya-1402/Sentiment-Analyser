import cv2
import numpy as np
from textblob import TextBlob
import speech_recognition as sr
import threading
import time

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces)

def listen_to_speech():
    global speech_text
    recognizer = sr.Recognizer()
    
    while True:
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
            
        try:
            text = recognizer.recognize_google(audio)
            speech_text = text
            print(f"Recognized: {text}")
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")

speech_text = ""
sentiment_result = "Neutral"

# Start the speech recognition thread
speech_thread = threading.Thread(target=listen_to_speech)
speech_thread.daemon = True
speech_thread.start()

# Initialize video capture
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces
    face_count = detect_faces(frame)
    
    # Analyze sentiment of speech
    if speech_text:
        sentiment_result = analyze_sentiment(speech_text)
        speech_text = ""  # Reset speech text after analysis
    
    # Display results on frame
    cv2.putText(frame, f"Faces: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Sentiment: {sentiment_result}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('Sentiment Analyser', frame)
    print(sentiment_result)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
