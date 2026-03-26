import cv2
import math
import numpy as np
from joblib import load
import os

def create_trackbar(trackbar_name, window_name, slider_max, initial_value=1):
    cv2.createTrackbar(trackbar_name, window_name, initial_value, slider_max, on_trackbar)

def on_trackbar(val):
    pass

def get_trackbar_value(trackbar_name, window_name):
    return int(cv2.getTrackbarPos(trackbar_name, window_name))

WINDOW_NAME = 'WINDOW'
TRACKBAR_THRESH_NAME  = 'Threshold'
TRACKBAR_THRESH_SLIDER_MAX = 255
TRACKBAR_KERNEL_NAME  = 'Kernel size'
TRACKBAR_KERNEL_SLIDER_MAX = 10
TRACKBAR_AREA_NAME    = 'Min area'
TRACKBAR_AREA_SLIDER_MAX = 2000
TRACKBAR_AREA_DEFAULT = 100
TRACKBAR_MAX_AREA_NAME    = 'Max area'
TRACKBAR_MAX_AREA_SLIDER_MAX = 50000
TRACKBAR_MAX_AREA_DEFAULT    = 10000
TRACKBAR_SOLIDITY_NAME    = 'Solidity x100'
TRACKBAR_SOLIDITY_SLIDER_MAX = 100
TRACKBAR_SOLIDITY_DEFAULT    = 50  # 0.50

LABELS = {1: 'spade', 2: 'heart', 3: 'diamond', 4: 'club'}

def get_hu_moments(contour):
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments)
    for i in range(7):
        hu[i] = -1 * math.copysign(1.0, hu[i]) * math.log10(abs(hu[i]) + 1e-10)
    return hu.flatten()

def get_solidity(contour):
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    return area / hull_area if hull_area > 0 else 0

def draw_trackbar_labels(frame):
    y_offset = 20
    line_height = 25
    thresh_val = get_trackbar_value(TRACKBAR_THRESH_NAME, WINDOW_NAME)
    cv2.putText(frame, f"Threshold: {thresh_val}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    kernel_val = get_trackbar_value(TRACKBAR_KERNEL_NAME, WINDOW_NAME)
    cv2.putText(frame, f"Kernel: {kernel_val}", (10, y_offset + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    area_val = get_trackbar_value(TRACKBAR_AREA_NAME, WINDOW_NAME)
    cv2.putText(frame, f"Min Area: {area_val}", (10, y_offset + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    max_area_val = get_trackbar_value(TRACKBAR_MAX_AREA_NAME, WINDOW_NAME)
    cv2.putText(frame, f"Max Area: {max_area_val}", (10, y_offset + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    solidity_val = get_trackbar_value(TRACKBAR_SOLIDITY_NAME, WINDOW_NAME) / 100.0
    cv2.putText(frame, f"Solidity: {solidity_val:.2f}", (10, y_offset + line_height * 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Presiona 'q' para salir", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    clf = load(os.path.join(BASE_DIR, 'machine', 'generated-files', 'model.joblib'))
    print("Modelo cargado")

    cv2.namedWindow(WINDOW_NAME)
    create_trackbar(TRACKBAR_THRESH_NAME,    WINDOW_NAME, TRACKBAR_THRESH_SLIDER_MAX, 100)
    create_trackbar(TRACKBAR_KERNEL_NAME,    WINDOW_NAME, TRACKBAR_KERNEL_SLIDER_MAX)
    create_trackbar(TRACKBAR_AREA_NAME,      WINDOW_NAME, TRACKBAR_AREA_SLIDER_MAX, TRACKBAR_AREA_DEFAULT)
    create_trackbar(TRACKBAR_MAX_AREA_NAME,  WINDOW_NAME, TRACKBAR_MAX_AREA_SLIDER_MAX, TRACKBAR_MAX_AREA_DEFAULT)
    create_trackbar(TRACKBAR_SOLIDITY_NAME,  WINDOW_NAME, TRACKBAR_SOLIDITY_SLIDER_MAX, TRACKBAR_SOLIDITY_DEFAULT)

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        cv2.imshow(WINDOW_NAME, frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        thresh_val = get_trackbar_value(TRACKBAR_THRESH_NAME, WINDOW_NAME)
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow('binary', thresh)

        kernel_size = max(1, get_trackbar_value(TRACKBAR_KERNEL_NAME, WINDOW_NAME))
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # 5. Filtrar por área y solidity
        min_area     = get_trackbar_value(TRACKBAR_AREA_NAME, WINDOW_NAME)
        max_area     = get_trackbar_value(TRACKBAR_MAX_AREA_NAME, WINDOW_NAME)
        min_solidity = get_trackbar_value(TRACKBAR_SOLIDITY_NAME, WINDOW_NAME) / 100.0

        filtered_contours = [
            c for c in contours
            if min_area < cv2.contourArea(c) < max_area
            and get_solidity(c) > min_solidity
        ]

        # 6. Clasificar con el modelo
        for contour in filtered_contours:
            hu = get_hu_moments(contour)
            sample = np.array(hu, dtype=np.float32).reshape(1, -1)
            predicted = clf.predict(sample)[0]
            label_name = LABELS.get(predicted, '?')

            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(frame, label_name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 7. Imagen anotada
        draw_trackbar_labels(frame)
        cv2.imshow("Clasificador", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()