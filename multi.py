import cv2 as cv
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

args = parser.parse_args()

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

inWidth = args.width
inHeight = args.height

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

# Use camera or video input
url = "http://192.168.0.109:8080/video"
cap = cv.VideoCapture(0)

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    height, width = out.shape[2], out.shape[3]

    all_points = []

    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = int((frameWidth * point[0]) / width)
        y = int((frameHeight * point[1]) / height)

        if conf > args.thr:
            all_points.append((x, y, conf))  # Store (x, y, confidence)
        else:
            all_points.append(None)

    # **Group keypoints into separate persons**
    persons = []
    used_points = set()

    for i, point in enumerate(all_points):
        if point is None or i in used_points:
            continue

        x, y, conf = point
        person = {i: (x, y)}

        # Check if other points belong to this person
        for j in range(len(all_points)):
            if j in person or all_points[j] is None:
                continue

            x2, y2, _ = all_points[j]

            # **Check if points are spatially close (same person)**
            if abs(x2 - x) < 100 and abs(y2 - y) < 100:  # Adjust threshold
                person[j] = (x2, y2)
                used_points.add(j)

        persons.append(person)

    # **Draw skeleton for each detected person**
    for person in persons:
        for pair in POSE_PAIRS:
            partFrom, partTo = pair
            idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]

            if idFrom in person and idTo in person:
                cv.line(frame, person[idFrom], person[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, person[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, person[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    # **Fall Detection Logic**
    for person in persons:
        if BODY_PARTS["Neck"] in person and BODY_PARTS["RHip"] in person and BODY_PARTS["LHip"] in person:
            neck_y = person[BODY_PARTS["Neck"]][1]
            hip_y = (person[BODY_PARTS["RHip"]][1] + person[BODY_PARTS["LHip"]][1]) / 2

            if hip_y - neck_y < 100:
                cv.putText(frame, "FALL DETECTED!", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow('Fall Detection using OpenPose', frame)
