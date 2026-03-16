import numpy as np
import os
import glob
import cv2
from math import copysign, log10


def save_hu_moments_to_file(hu_moments, filepath='tps/tp2/hu_moments/hu_moments.txt'):
    """
    Guarda los momentos de Hu en un archivo.
    Cada línea contiene los 7 valores de los momentos de Hu separados por espacios.
    """
    # Crear el directorio si no existe
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convertir hu_moments a un array plano y guardarlo como texto
    hu_array = hu_moments.flatten() if isinstance(hu_moments, np.ndarray) else np.array(hu_moments).flatten()
    with open(filepath, 'a') as f:
        f.write(' '.join(map(str, hu_array)) + '\n')


def load_hu_moments_from_file(filepath='tps/tp2/hu_moments/hu_moments.txt'):
    """
    Carga los momentos de Hu desde un archivo y los devuelve como una lista de arrays numpy.
    Cada línea del archivo contiene 7 valores separados por espacios.
    """
    saved_hu_moments = []
    if not os.path.exists(filepath):
        return saved_hu_moments
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                values = list(map(float, line.split()))
                if len(values) == 7:
                    # Convertir a array numpy con la misma forma que cv2.HuMoments devuelve
                    hu_moments = np.array(values, dtype=np.float32).reshape(7, 1)
                    saved_hu_moments.append(hu_moments)
    
    return saved_hu_moments


def get_hu_moments(contour):
    """
    Calcula los momentos de Hu para un contorno dado.
    Maneja el caso especial cuando el valor es cero o muy cercano a cero.
    """
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments)
    for i in range(len(hu_moments)):
        value = float(hu_moments[i][0])  # Extract scalar from array
        abs_value = abs(value)
        # Evitar log10(0) usando un valor epsilon muy pequeño
        if abs_value < 1e-10:
            hu_moments[i] = 0.0
        else:
            hu_moments[i] = -1 * copysign(1.0, value) * log10(abs_value)
    return hu_moments


def process_and_save_images(imgs_folder='tps/tp2/hu_moments/imgs', output_file='tps/tp2/hu_moments/hu_moments.txt'):
    """
    Procesa todas las imágenes en la carpeta especificada, calcula sus momentos de Hu
    y los guarda en el archivo de salida.
    """
    # Limpiar el archivo de salida si existe
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Buscar todas las imágenes en la carpeta
    image_extensions = ['*.jpeg', '*.jpg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(imgs_folder, ext)))
        image_files.extend(glob.glob(os.path.join(imgs_folder, ext.upper())))
    
    if not image_files:
        print(f"No se encontraron imágenes en {imgs_folder}")
        return
    
    print(f"Procesando {len(image_files)} imágenes...")
    
    for img_path in image_files:
        print(f"Procesando: {img_path}")
        # Leer la imagen
        image = cv2.imread(img_path)
        if image is None:
            print(f"  Error: No se pudo leer {img_path}")
            continue
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar threshold
        _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        
        # Invertir si es necesario (para que las figuras sean blancas sobre fondo negro)
        thresh = 255 - thresh
        
        # Encontrar contornos
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            print(f"  Advertencia: No se encontraron contornos en {img_path}")
            continue
        
        # Obtener el contorno de mayor área
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calcular momentos de Hu
        hu_moments = get_hu_moments(largest_contour)
        
        # Guardar los momentos de Hu
        save_hu_moments_to_file(hu_moments, output_file)
        print(f"  ✓ Momentos de Hu guardados para {os.path.basename(img_path)}")
    
    print(f"\n✓ Proceso completado. Momentos de Hu guardados en {output_file}")
