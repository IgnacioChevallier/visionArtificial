import cv2
from math import copysign, log10

def create_trackbar(trackbar_name, window_name, slider_max, initial_value=1):
    cv2.createTrackbar(trackbar_name, window_name, initial_value, slider_max, on_trackbar)

def on_trackbar(val):
    pass

def get_trackbar_value(trackbar_name, window_name):
    return int(cv2.getTrackbarPos(trackbar_name, window_name))


WINDOW_NAME = 'WINDOW'

TRACKBAR_THRESH_NAME = 'Threshold'
TRACKBAR_THRESH_SLIDER_MAX = 255

TRACKBAR_KERNEL_NAME = 'Kernel size'
TRACKBAR_KERNEL_SLIDER_MAX = 10

TRACKBAR_AREA_NAME = 'Min area'
TRACKBAR_AREA_SLIDER_MAX = 2000
TRACKBAR_AREA_DEFAULT = 100


def main():
    cv2.namedWindow(WINDOW_NAME)
    create_trackbar(TRACKBAR_THRESH_NAME, WINDOW_NAME, TRACKBAR_THRESH_SLIDER_MAX)
    create_trackbar(TRACKBAR_KERNEL_NAME, WINDOW_NAME, TRACKBAR_KERNEL_SLIDER_MAX)
    create_trackbar(TRACKBAR_AREA_NAME, WINDOW_NAME, TRACKBAR_AREA_SLIDER_MAX, TRACKBAR_AREA_DEFAULT)
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        cv2.imshow(WINDOW_NAME, frame)
        
        # 1. Convertir la imagen a monocromática
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', gray)

        # 2. Aplicar un threshold con umbral ajustable con una barra de desplazamiento
        trackbar_thresh_value = get_trackbar_value(TRACKBAR_THRESH_NAME, WINDOW_NAME)
        _, thresh = cv2.threshold(gray, trackbar_thresh_value, 255, cv2.THRESH_BINARY)
        # cv2.imshow('binary', thresh)

        # 3. Aplicar operaciones morfológicas para eliminar ruido de la imagen
        kernel_size_value = get_trackbar_value(TRACKBAR_KERNEL_NAME, WINDOW_NAME)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size_value, kernel_size_value))
        
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # cv2.imshow('opening', opening)

        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('closing', closing)

        # 4. Obtener varios contornos en una misma imagen
        ret1, thresh1 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # 5. Filtrar contornos que se pueden descartar de antemano
        min_area = get_trackbar_value(TRACKBAR_AREA_NAME, WINDOW_NAME)
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        cv2.drawContours(frame, filtered_contours, -1, (100, 0, 100), 2)
        cv2.imshow("Contornos patente", frame)

        # 6. Compara cada contorno con todos los objetos de referencia, usando matchShapes()
        saved_hu_moments = []
        for contour in filtered_contours:
            hu_moments = get_hu_moments(contour)
            if compare_hu_moments(hu_moments, saved_hu_moments, 0.1):
                cv2.show("Patente encontrada", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

def get_hu_moments(contour):
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments)
    for i in range(len(hu_moments)):
        hu_moments[i] = -1 * copysign(1.0, hu_moments[i]) * log10(abs(hu_moments[i]))
    return hu_moments

def compare_hu_moments(hu_moments, saved_hu_moments, max_diff):
    for moments in saved_hu_moments:
        if cv2.matchShapes(hu_moments, moments, cv2.CONTOURS_MATCH_I2, 0) < max_diff:
            return True
    return False
    
main()
