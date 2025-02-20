import cv2 as cv
import numpy as np
import time
import math

# Load OpenPose model
net = cv.dnn.readNetFromCaffe("pose_deploy_linevec.prototxt", "pose_iter_440000.caffemodel")
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Body parts mapping
BODY_PARTS = {"Neck": 1, "RHip": 8, "LHip": 11}

def calculate_angle(p1, p2):
    return abs(math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])))

# Initialize video capture
cap = cv.VideoCapture(0)
prev_y = {}
fall_start_time = {}
FALL_THRESHOLD_TIME = 2  # seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    inpBlob = cv.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(inpBlob)
    out = net.forward()
    
    people = []  # Store detected keypoints for multiple persons
    
    for i in range(out.shape[1]):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        
        if conf > 0.1:  # Confidence threshold
            x = int((frameWidth * point[0]) / out.shape[3])
            y = int((frameHeight * point[1]) / out.shape[2])
            people.append((x, y, conf))
    
    for i, person in enumerate(people):
        points = {bp: None for bp in BODY_PARTS}
        
        for part, idx in BODY_PARTS.items():
            if idx < len(people):
                points[part] = (people[idx][0], people[idx][1])
        
        fall_detected = False
        if points["Neck"] and points["RHip"] and points["LHip"]:
            neck = points["Neck"]
            hip = ((points["RHip"][0] + points["LHip"][0]) // 2, (points["RHip"][1] + points["LHip"][1]) // 2)
            angle = calculate_angle(neck, hip)
            
            if i in prev_y and abs(neck[1] - prev_y[i]) > 30:  # Check velocity
                if i not in fall_start_time:
                    fall_start_time[i] = time.time()
                
                elif time.time() - fall_start_time[i] > FALL_THRESHOLD_TIME:
                    fall_detected = True
                    cv.putText(frame, "Fall Detected!", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            prev_y[i] = neck[1]
    
    cv.imshow("Fall Detection", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()