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

# Video feed URL (replace with your own)
url = "http://192.168.97.217:8080/video"
cap = cv2.VideoCapture(url)

# For voice notifications
last_played_time = 0
message_shown = False
running = False


def fall_detection_logic(frame):
    """Detect fall based on neck and hip positions."""
    fall_detected = False
    inWidth, inHeight = 368, 368

    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frame.shape[1] * point[0]) / out.shape[3]
        y = (frame.shape[0] * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > 0.2 else None)

    if points[BODY_PARTS["Neck"]] and points[BODY_PARTS["RHip"]] and points[BODY_PARTS["LHip"]]:
        neck_y = points[BODY_PARTS["Neck"]][1]
        hip_y = (points[BODY_PARTS["RHip"]][1] + points[BODY_PARTS["LHip"]][1]) / 2
        if hip_y - neck_y < 100:
            fall_detected = True
            # print("Fall Detected!")

    return fall_detected, points


def detection_loop():
    global last_played_time, running, message_shown
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            continue

        # Fall detection
        fall_detected, points = fall_detection_logic(frame)

        if fall_detected and not message_shown:
            # Play alert sound
            current_time = time.time()
            if current_time - last_played_time >= 5:
                pygame.mixer.music.load('danger.mp3')  # Provide your own audio file path
                pygame.mixer.music.play()
                last_played_time = current_time

            # Show fall detection message on GUI
            show_message("Fall Detected!", 3)
            message_shown = True

        # Draw detected pose on frame
        for pair in POSE_PAIRS:
            partFrom, partTo = pair
            idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]
            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(image=img)
        panel.config(image=img_tk)
        panel.img_tk = img_tk
        panel.update_idletasks()
        root.update()  # Force GUI update


def show_message(message, duration):
    """Show a message on the GUI for a specified duration."""
    message_label.config(text=message)  # Update label with the message
    root.after(duration * 1000, clear_message)  # Call clear_message after 'duration' seconds


def clear_message():
    """Clear the message on the GUI."""
    global message_shown
    message_label.config(text="")  # Clear the message label
    message_shown = False  # Reset the flag after the message is cleared


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
root.title("Fall Detection System")

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
