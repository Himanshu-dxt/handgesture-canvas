import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize the arrays to handle color points of different colors for both hands
bpoints = [deque(maxlen=1024), deque(maxlen=1024)]
gpoints = [deque(maxlen=1024), deque(maxlen=1024)]
rpoints = [deque(maxlen=1024), deque(maxlen=1024)]
ypoints = [deque(maxlen=1024), deque(maxlen=1024)]

# These indexes will be used to mark the points in particular arrays of specific colors for both hands
blue_index = [0, 0]
green_index = [0, 0]
red_index = [0, 0]
yellow_index = [0, 0]

# The kernel to be used for dilation purposes
kernel = np.ones((5, 5), np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = [0, 0]

# Initialize the canvas setup
paintWindow = np.zeros((471, 636, 3)) + 255
paintWindow = cv2.circle(paintWindow, (90, 33), 30, (0, 0, 0), -1)
paintWindow = cv2.circle(paintWindow, (208, 33), 30, (255, 0, 0), -1)
paintWindow = cv2.circle(paintWindow, (318, 33), 30, (0, 255, 0), -1)
paintWindow = cv2.circle(paintWindow, (428, 33), 30, (0, 0, 255), -1)
paintWindow = cv2.circle(paintWindow, (538, 33), 30, (0, 255, 255), -1)

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize Mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)
ret = True
while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.circle(frame, (90, 33), 30, (0, 0, 0), -1)
    frame = cv2.circle(frame, (208, 33), 30, (255, 0, 0), -1)
    frame = cv2.circle(frame, (318, 33), 30, (0, 255, 0), -1)
    frame = cv2.circle(frame, (428, 33), 30, (0, 0, 255), -1)
    frame = cv2.circle(frame, (538, 33), 30, (0, 255, 255), -1)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # Post process the result
    if result.multi_hand_landmarks:
        for hand_idx, handslms in enumerate(result.multi_hand_landmarks):
            landmarks = []
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            fore_finger = (landmarks[8][0], landmarks[8][1])
            center = fore_finger
            thumb = (landmarks[4][0], landmarks[4][1])
            cv2.circle(frame, center, 3, (0, 255, 0), -1)
            if (thumb[1] - center[1] < 30):
                bpoints[hand_idx].append(deque(maxlen=512))
                blue_index[hand_idx] += 1
                gpoints[hand_idx].append(deque(maxlen=512))
                green_index[hand_idx] += 1
                rpoints[hand_idx].append(deque(maxlen=512))
                red_index[hand_idx] += 1
                ypoints[hand_idx].append(deque(maxlen=512))
                yellow_index[hand_idx] += 1

            elif center[1] <= 65:
                if ((center[0] - 90) ** 2 + (center[1] - 33) ** 2) <= 30 ** 2:  # Clear Button
                    bpoints = [deque(maxlen=512), deque(maxlen=512)]
                    gpoints = [deque(maxlen=512), deque(maxlen=512)]
                    rpoints = [deque(maxlen=512), deque(maxlen=512)]
                    ypoints = [deque(maxlen=512), deque(maxlen=512)]

                    blue_index = [0, 0]
                    green_index = [0, 0]
                    red_index = [0, 0]
                    yellow_index = [0, 0]

                    paintWindow[67:, :, :] = 255
                elif ((center[0] - 208) ** 2 + (center[1] - 33) ** 2) <= 30 ** 2:
                    colorIndex[hand_idx] = 0  # Blue
                elif ((center[0] - 318) ** 2 + (center[1] - 33) ** 2) <= 30 ** 2:
                    colorIndex[hand_idx] = 1  # Green
                elif ((center[0] - 428) ** 2 + (center[1] - 33) ** 2) <= 30 ** 2:
                    colorIndex[hand_idx] = 2  # Red
                elif ((center[0] - 538) ** 2 + (center[1] - 33) ** 2) <= 30 ** 2:
                    colorIndex[hand_idx] = 3  # Yellow
            else:
                if colorIndex[hand_idx] == 0:
                    if blue_index[hand_idx] >= len(bpoints[hand_idx]):
                        bpoints[hand_idx].append(deque(maxlen=512))
                    bpoints[hand_idx][blue_index[hand_idx]].appendleft(center)
                elif colorIndex[hand_idx] == 1:
                    if green_index[hand_idx] >= len(gpoints[hand_idx]):
                        gpoints[hand_idx].append(deque(maxlen=512))
                    gpoints[hand_idx][green_index[hand_idx]].appendleft(center)
                elif colorIndex[hand_idx] == 2:
                    if red_index[hand_idx] >= len(rpoints[hand_idx]):
                        rpoints[hand_idx].append(deque(maxlen=512))
                    rpoints[hand_idx][red_index[hand_idx]].appendleft(center)
                elif colorIndex[hand_idx] == 3:
                    if yellow_index[hand_idx] >= len(ypoints[hand_idx]):
                        ypoints[hand_idx].append(deque(maxlen=512))
                    ypoints[hand_idx][yellow_index[hand_idx]].appendleft(center)
    else:
        for i in range(2):
            bpoints[i].append(deque(maxlen=512))
            blue_index[i] += 1
            gpoints[i].append(deque(maxlen=512))
            green_index[i] += 1
            rpoints[i].append(deque(maxlen=512))
            red_index[i] += 1
            ypoints[i].append(deque(maxlen=512))
            yellow_index[i] += 1

    # Draw lines of all the colors on the canvas and frame
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(2):
        for j in range(len(points)):
            for k in range(len(points[j][i])):
                for l in range(1, len(points[j][i][k])):
                    if points[j][i][k][l - 1] is None or points[j][i][k][l] is None:
                        continue
                    cv2.line(frame, points[j][i][k][l - 1], points[j][i][k][l], colors[j], 2)
                    cv2.line(paintWindow, points[j][i][k][l - 1], points[j][i][k][l], colors[j], 2)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()