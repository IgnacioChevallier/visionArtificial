import cv2
import numpy as np
from momentizer import load_hu_moments_from_file, get_hu_moments, process_and_save_images

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

TRACKBAR_MATCH_NAME = 'Match threshold'
TRACKBAR_MATCH_SLIDER_MAX = 500  # 0.0 to 5.0 (scaled by 100)
TRACKBAR_MATCH_DEFAULT = 150  # 1.5 default


def draw_trackbar_labels(frame):
    """Dibuja etiquetas explicativas para cada trackbar en la imagen."""
    y_offset = 20
    line_height = 25
    
    # Threshold label
    thresh_val = get_trackbar_value(TRACKBAR_THRESH_NAME, WINDOW_NAME)
    cv2.putText(frame, f"Threshold: {thresh_val} (umbral binarizacion)", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Kernel size label
    kernel_val = get_trackbar_value(TRACKBAR_KERNEL_NAME, WINDOW_NAME)
    cv2.putText(frame, f"Kernel: {kernel_val} (tamano morfologia)", 
               (10, y_offset + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Min area label
    area_val = get_trackbar_value(TRACKBAR_AREA_NAME, WINDOW_NAME)
    cv2.putText(frame, f"Min Area: {area_val} (filtro contornos)", 
               (10, y_offset + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Match threshold label
    match_val = get_trackbar_value(TRACKBAR_MATCH_NAME, WINDOW_NAME) / 100.0
    cv2.putText(frame, f"Match Threshold: {match_val:.2f} (sensibilidad match)", 
               (10, y_offset + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Help text
    cv2.putText(frame, "Presiona 'q' para salir, 'd' para debug", 
               (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

def main():
    cv2.namedWindow(WINDOW_NAME)
    create_trackbar(TRACKBAR_THRESH_NAME, WINDOW_NAME, TRACKBAR_THRESH_SLIDER_MAX, 100)
    create_trackbar(TRACKBAR_KERNEL_NAME, WINDOW_NAME, TRACKBAR_KERNEL_SLIDER_MAX)
    create_trackbar(TRACKBAR_AREA_NAME, WINDOW_NAME, TRACKBAR_AREA_SLIDER_MAX, TRACKBAR_AREA_DEFAULT)
    create_trackbar(TRACKBAR_MATCH_NAME, WINDOW_NAME, TRACKBAR_MATCH_SLIDER_MAX, TRACKBAR_MATCH_DEFAULT)
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
        cv2.imshow('binary', thresh)

        # 3. Aplicar operaciones morfológicas para eliminar ruido de la imagen
        kernel_size_value = get_trackbar_value(TRACKBAR_KERNEL_NAME, WINDOW_NAME)
        # Asegurar que kernel_size sea al menos 1
        kernel_size_value = max(1, kernel_size_value)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size_value, kernel_size_value))
        
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # cv2.imshow('opening', opening)

        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('closing', closing)

        # 4. Obtener varios contornos en una misma imagen
        contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # 5. Filtrar contornos que se pueden descartar de antemano
        min_area = get_trackbar_value(TRACKBAR_AREA_NAME, WINDOW_NAME)
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        # 6. Compara cada contorno con todos los objetos de referencia, usando matchShapes()
        saved_hu_moments = load_hu_moments_from_file()
        match_threshold = get_trackbar_value(TRACKBAR_MATCH_NAME, WINDOW_NAME) / 100.0  # Convertir a decimal (0.0-5.0)
        
        match_found = False
        match_count = 0
        
        for contour in filtered_contours:
            hu_moments = get_hu_moments(contour)
            # Usar debug solo si presionas 'd'
            match_result = compare_hu_moments(hu_moments, saved_hu_moments, match_threshold)
            if match_result:
                match_found = True
                match_count += 1
                # Dibujar el contorno en verde cuando hay match
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
                # Agregar texto indicando el match
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(frame, "MATCH!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Dibujar todos los contornos (en magenta si no hay match, verde si hay match)
        if not match_found:
            cv2.drawContours(frame, filtered_contours, -1, (100, 0, 100), 2)
        
        # Dibujar etiquetas de los trackbars
        draw_trackbar_labels(frame)
        
        # Mostrar información de matches
        if match_found:
            cv2.putText(frame, f"Matches encontrados: {match_count}", (10, frame.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Contornos patente", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

# def compare_hu_moments(hu_moments, saved_hu_moments, max_diff):
#     """
#     Compara los momentos de Hu con los guardados.
#     Retorna True si encuentra un match dentro del umbral.
#     """
#     if not saved_hu_moments:
#         return False
#
#     best_match_score = float('inf')
#     for i, moments in enumerate(saved_hu_moments):
#         match_score = cv2.matchShapes(hu_moments, moments, cv2.CONTOURS_MATCH_I2, 0)
#         print(match_score)
#         if match_score < best_match_score:
#             best_match_score = match_score
#     return best_match_score < max_diff

def compare_hu_moments(hu_moments, saved_hu_moments, max_diff):
    if not saved_hu_moments:
        return False

    best_match_score = float('inf')
    for moments in saved_hu_moments:
        # Comparar los 7 valores directamente
        diff = np.sum(np.abs(hu_moments.flatten() - moments.flatten()))
        if diff < best_match_score:
            best_match_score = diff

    return best_match_score < max_diff

if __name__ == '__main__':
    # Procesar imágenes y guardar sus momentos de Hu
    process_and_save_images()
    
    # Ejecutar el programa principal
    main()
