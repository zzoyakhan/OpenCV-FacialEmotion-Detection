import cv2
from deepface import DeepFace

# Load the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier('/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture image")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces and analyze emotions
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]  # Extract the face region
        
        try:
            # Perform emotion detection using DeepFace
            analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
        except Exception as e:
            emotion = "Unknown"
        
        # Draw the rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Display the detected emotion
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame with detected faces and emotions
    cv2.imshow('Facial & Emotion Detection', frame)

    # Check for key press; break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
