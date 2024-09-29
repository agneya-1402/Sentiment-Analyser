import speech_recognition as sr
from textblob import TextBlob
import pyaudio

def record_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak now...")
        audio = r.listen(source)
    return audio

def speech_to_text(audio):
    r = sr.Recognizer()
    try:
        text = r.recognize_google(audio)
        print("You said: " + text)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    return None

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

def main():
    while True:
        audio = record_audio()
        text = speech_to_text(audio)
        if text:
            sentiment = analyze_sentiment(text)
            print(f"Sentiment: {sentiment}")
        
        choice = input("Press Enter to analyze another sentence or 'q' to quit: ")
        if choice.lower() == 'q':
            break

if __name__ == "__main__":
    main()
    
