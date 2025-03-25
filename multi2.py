import cv2
import numpy as np
import pygame
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

pygame.mixer.init()

# Load the pre-trained model for pose detection
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

# Define body parts and pose pairs for detecting fall
BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}
POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

# Threshold for confidence and person detection
CONFIDENCE_THRESHOLD = 0.2
PERSON_PROXIMITY_THRESHOLD = 100

# Video feed settings
cap = cv2.VideoCapture(0)  # Change to your preferred video source

# For voice notifications
last_played_time = 0
message_shown = False
running = False


def detect_persons(frame, out):
    """Detect and group keypoints into separate persons."""
    height, width = out.shape[2], out.shape[3]
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    
    all_points = []
    
    # Detect keypoints
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = int((frameWidth * point[0]) / width)
        y = int((frameHeight * point[1]) / height)

        if conf > CONFIDENCE_THRESHOLD:
            all_points.append((x, y, conf, i))
        else:
            all_points.append(None)
    
    # Group keypoints into persons
    persons = []
    used_points = set()

    for i, point in enumerate(all_points):
        if point is None or i in used_points:
            continue

        x, y, conf, part_id = point
        person = {part_id: (x, y)}
        used_points.add(i)

        # Find other points belonging to this person
        for j, other_point in enumerate(all_points):
            if j in person or other_point is None or j in used_points:
                continue

            x2, y2, _, _ = other_point
            
            # Check if points are spatially close
            if abs(x2 - x) < PERSON_PROXIMITY_THRESHOLD and abs(y2 - y) < PERSON_PROXIMITY_THRESHOLD:
                person[j] = (x2, y2)
                used_points.add(j)

        persons.append(person)
    
    return persons


def detect_fall(person):
    """Detect fall for a single person."""
    if (BODY_PARTS["Neck"] in person and 
        BODY_PARTS["RHip"] in person and 
        BODY_PARTS["LHip"] in person):
        
        neck_y = person[BODY_PARTS["Neck"]][1]
        hip_y = (person[BODY_PARTS["RHip"]][1] + person[BODY_PARTS["LHip"]][1]) / 2
        
        return hip_y - neck_y < 100
    
    return False


def detection_loop():
    global last_played_time, running, message_shown
    
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            continue

        # Prepare frame for detection
        inWidth, inHeight = 368, 368
        net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), 
                                           (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]

        # Detect persons
        persons = detect_persons(frame, out)
        
        # Fall detection and visualization
        fall_detected = False
        for person in persons:
            # Draw skeleton for each person
            for pair in POSE_PAIRS:
                partFrom, partTo = pair
                idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]

                if idFrom in person and idTo in person:
                    cv2.line(frame, person[idFrom], person[idTo], (0, 255, 0), 3)
                    cv2.ellipse(frame, person[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                    cv2.ellipse(frame, person[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            
            # Check for fall
            if detect_fall(person):
                fall_detected = True
                cv2.putText(frame, "FALL DETECTED!", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Alert and message handling
        if fall_detected and not message_shown:
            current_time = time.time()
            if current_time - last_played_time >= 5:
                pygame.mixer.music.load('danger.mp3')  # Provide your own audio file path
                pygame.mixer.music.play()
                last_played_time = current_time

            show_message("Fall Detected!", 3)
            message_shown = True

        # Update GUI
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(image=img)
        panel.config(image=img_tk)
        panel.img_tk = img_tk
        panel.update_idletasks()
        root.update()


def show_message(message, duration):
    """Show a message on the GUI for a specified duration."""
    message_label.config(text=message)
    root.after(duration * 1000, clear_message)


def clear_message():
    """Clear the message on the GUI."""
    global message_shown
    message_label.config(text="")
    message_shown = False


def start_detection():
    """Start the detection process."""
    global running
    running = True
    threading.Thread(target=detection_loop, daemon=True).start()


def stop_detection():
    """Stop the detection process."""
    global running
    running = False
    cap.release()
    cv2.destroyAllWindows()
    root.quit()


# Tkinter GUI setup
root = tk.Tk()
root.title("Multi-Person Fall Detection System")

# Panel for displaying video feed
panel = tk.Label(root)
panel.pack(padx=10, pady=10)

# Message Label for displaying detection messages
message_label = tk.Label(root, text="", font=("Helvetica", 16), fg="red")
message_label.pack(padx=10, pady=10)

# Start and Stop buttons
start_button = ttk.Button(root, text="Start Detection", command=start_detection)
start_button.pack(side=tk.LEFT, padx=10, pady=10)

stop_button = ttk.Button(root, text="Stop Detection", command=stop_detection)
stop_button.pack(side=tk.RIGHT, padx=10, pady=10)

root.mainloop()