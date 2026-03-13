import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

RED_COLOR = (0, 0, 255)
capture = cv2.VideoCapture(0)


def calculate_distance(p1, p2):
    """Calculate distance between two points"""
    return np.sqrt((p1.x - p2.x) * 2 + (p1.y - p2.y) * 2)


def finger_counter(hand_landmarks):

    fingers = []

    # index
    if hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y < \
       hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # middle
    if hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y < \
       hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # ring
    if hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].y < \
       hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # pinky
    if hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].y < \
       hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # thumb (distinto porque se mueve lateralmente)
    if hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].x > \
       hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].x:
        fingers.append(1)
    else:
        fingers.append(0)

    return sum(fingers)


while capture.isOpened():
    ret, frame = capture.read()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    right_hand_count = 0
    left_hand_count = 0
    
    # Check for right hand only
    if results.right_hand_landmarks:
        right_hand_count = finger_counter(results.right_hand_landmarks)
    # Check for left hand only
    if results.left_hand_landmarks:
        left_hand_count = finger_counter(results.left_hand_landmarks)

    print(f"Right hand fingers: {right_hand_count}, Left hand fingers: {left_hand_count}, Brightness: {(right_hand_count + left_hand_count) * 10}%")
    sbc.set_brightness((right_hand_count + left_hand_count) * 10)

    cv2.imshow("Image", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()