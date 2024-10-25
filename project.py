import cv2
import face_recognition
import numpy as np
import pandas as pd
import smtplib
from datetime import datetime
import os

# Load known faces
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir(known_faces_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])  # use filename as name
            
    return known_face_encodings, known_face_names

# Send notification email
def send_email(name):
    sender_email = "your_email@example.com"
    receiver_email = "recipient_email@example.com"
    password = "your_password"
    
    message = f"""Subject: Attendance Notification\n\nDear {name},\n\nThis is to notify you that you were absent today."""
    
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

# Main attendance tracking function
def track_attendance(known_faces_dir):
    known_face_encodings, known_face_names = load_known_faces(known_faces_dir)
    attendance = pd.DataFrame(columns=["Name", "Date"])
    
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        names_in_frame = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            names_in_frame.append(name)

        # Record attendance
        for name in names_in_frame:
            if name not in attendance["Name"].values:
                attendance = attendance.append({"Name": name, "Date": datetime.now().date()}, ignore_index=True)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, names_in_frame):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Video', frame)

        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    # Check for absentees and notify
    all_names = set(known_face_names)
    present_names = set(attendance["Name"].values)

    absentees = all_names - present_names
    for absentee in absentees:
        send_email(absentee)

    # Save attendance to CSV
    attendance.to_csv("attendance.csv", index=False)

if __name__ == "__main__":
    track_attendance("path_to_known_faces_directory")