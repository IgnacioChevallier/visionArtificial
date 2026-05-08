import os
import sys

# OpenCV 4.10+ puede probar el backend Orbbec antes que V4L2 y ensuciar la salida.
# Conviene definir esto antes de importar cv2.
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_OBSENSOR", "0")

import cv2
import numpy as np


WINDOW_CAMERA = "camara"
WINDOW_FRONT = "vista frontal"
GRID_CELLS = 3
OUTPUT_SIZE = 500


def destroy_window_if_exists(name):
    """Cierra una ventana de OpenCV si existe."""
    try:
        cv2.destroyWindow(name)
    except cv2.error:
        pass


def _capture_delivers_frame(cap, attempts=45):
    """isOpened() no garantiza imagen; exigimos al menos un frame valido."""
    for _ in range(attempts):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return True
    return False


def _try_open_capture_index(index):
    """Prueba abrir una camara por indice usando V4L2 y luego el backend por defecto."""
    v4l2 = getattr(cv2, "CAP_V4L2", None)
    dev_path = f"/dev/video{int(index)}"
    tries = []

    if v4l2 is not None and os.path.exists(dev_path):
        tries.append((dev_path, v4l2))
    if v4l2 is not None:
        tries.append((index, v4l2))
    tries.append((index, None))

    for source, backend in tries:
        cap = cv2.VideoCapture(source, backend) if backend is not None else cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            continue
        if _capture_delivers_frame(cap):
            return cap
        cap.release()
    return None


def open_video_capture():
    """
    Abre la fuente de video.

    Variables utiles:
    - HOMOGRAFIA_VIDEO=/ruta/video.mp4 o /dev/videoN para usar una fuente concreta.
    - HOMOGRAFIA_CAMERA=N para elegir un indice de camara.
    - HOMOGRAFIA_SCAN_LAST=N para decidir hasta que indice buscar si no se fijo camara.
    """
    path = (os.environ.get("HOMOGRAFIA_VIDEO") or "").strip()
    if path:
        cap = cv2.VideoCapture(path)
        if cap.isOpened() and _capture_delivers_frame(cap):
            return cap
        cap.release()
        return None

    fixed_index = os.environ.get("HOMOGRAFIA_CAMERA") or os.environ.get("OPENCV_CAMERA_INDEX")
    if fixed_index is not None:
        indices = [int(fixed_index)]
    else:
        last = max(0, int(os.environ.get("HOMOGRAFIA_SCAN_LAST", "4")))
        indices = list(range(last + 1))

    for index in indices:
        cap = _try_open_capture_index(index)
        if cap is not None:
            if len(indices) > 1:
                print(f"Camara encontrada en indice {index}")
            return cap
    return None


def camera_open_fail_message():
    path = (os.environ.get("HOMOGRAFIA_VIDEO") or "").strip()
    if path:
        return f"No se pudo abrir HOMOGRAFIA_VIDEO={path!r}.\n"
    return (
        "No se pudo abrir ninguna camara.\n"
        "Proba, por ejemplo: HOMOGRAFIA_CAMERA=0 python homografia.py\n"
        "O con un video: HOMOGRAFIA_VIDEO=/ruta/al/video.mp4 python homografia.py\n"
    )


def order_points(points):
    """
    Ordena 4 puntos como: arriba-izquierda, arriba-derecha, abajo-derecha, abajo-izquierda.

    Este orden es el que espera cv2.getPerspectiveTransform para que el cuadrado frontal
    no salga rotado o espejado.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    sums = pts.sum(axis=1)
    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]

    diffs = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diffs)]
    rect[3] = pts[np.argmax(diffs)]
    return rect


def make_square_points(size=OUTPUT_SIZE):
    """Coordenadas conocidas del cuadrado visto de frente."""
    return np.array(
        [
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1],
        ],
        dtype=np.float32,
    )


def compute_homographies(image_points):
    """
    Calcula las dos matrices que necesitamos.

    image_to_front: lleva pixeles de la camara a la vista frontal.
    front_to_image: lleva puntos de la vista frontal a la imagen de la camara.

    Son transformaciones inversas conceptualmente, pero las calculamos ambas con OpenCV
    para mantener claro cual se usa en cada visualizacion.
    """
    image_rect = order_points(image_points)
    front_rect = make_square_points()

    image_to_front = cv2.getPerspectiveTransform(image_rect, front_rect)
    front_to_image = cv2.getPerspectiveTransform(front_rect, image_rect)
    return image_to_front, front_to_image, image_rect


def detect_qr_points(frame, detector):
    """Devuelve los 4 vertices del QR si se detecta uno, o None si no hay deteccion."""
    _text, points, _straight_qrcode = detector.detectAndDecode(frame)
    if points is None:
        return None
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) != 4:
        return None
    return points


def draw_points(frame, points, color=(0, 255, 255)):
    """Marca los vertices usados para computar la homografia."""
    for index, (x, y) in enumerate(np.asarray(points, dtype=np.int32), start=1):
        cv2.circle(frame, (int(x), int(y)), 6, color, -1)
        cv2.putText(
            frame,
            str(index),
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_grid(frame, front_to_image, cells=GRID_CELLS):
    """
    Dibuja una grilla cuadrada en perspectiva sobre la imagen original.

    La grilla se define facil en la vista frontal y despues se proyecta a la camara
    usando front_to_image.
    """
    size = OUTPUT_SIZE - 1
    line_color = (0, 255, 0)
    border_color = (0, 180, 255)

    for i in range(cells + 1):
        pos = i * size / cells
        front_lines = [
            np.array([[[pos, 0]], [[pos, size]]], dtype=np.float32),
            np.array([[[0, pos]], [[size, pos]]], dtype=np.float32),
        ]

        for line in front_lines:
            projected = cv2.perspectiveTransform(line, front_to_image).reshape(2, 2)
            p1 = tuple(np.round(projected[0]).astype(int))
            p2 = tuple(np.round(projected[1]).astype(int))
            color = border_color if i in (0, cells) else line_color
            thickness = 3 if i in (0, cells) else 2
            cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def put_status(frame, mode, has_homography, click_count):
    """Texto corto de ayuda para saber en que modo esta el programa."""
    if mode == "qr":
        text = "QR: cualquier tecla computa homografia"
        color = (0, 255, 255)
    elif mode == "manual":
        text = f"Manual: clics {click_count}/4 - cualquier tecla aborta"
        color = (0, 255, 255)
    else:
        status = "homografia lista" if has_homography else "sin homografia"
        text = f"q: QR | h: manual | ESC: salir | {status}"
        color = (255, 255, 255)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


class ManualPointCollector:
    """Guarda los puntos clickeados durante el modo de homografia asistida."""

    def __init__(self):
        self.enabled = False
        self.points = []

    def reset(self):
        self.points = []

    def start(self):
        self.enabled = True
        self.reset()

    def stop(self):
        self.enabled = False
        self.reset()

    def on_click(self, event, x, y, _flags, _param):
        if not self.enabled:
            return
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            print(f"Punto {len(self.points)}/4: ({x}, {y})")


def main():
    cap = open_video_capture()
    if cap is None:
        sys.stderr.write(camera_open_fail_message())
        sys.exit(1)

    qr_detector = cv2.QRCodeDetector()
    manual = ManualPointCollector()
    mode = "view"

    image_to_front = None
    front_to_image = None
    selected_points = None

    cv2.namedWindow(WINDOW_CAMERA)
    cv2.setMouseCallback(WINDOW_CAMERA, manual.on_click)

    print("Controles: q = detectar QR, h = marcar 4 puntos, ESC = salir.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            cv2.waitKey(30)
            continue

        display = frame.copy()

        if mode == "qr":
            qr_points = detect_qr_points(frame, qr_detector)
            if qr_points is not None:
                draw_points(display, order_points(qr_points))
        elif mode == "manual":
            draw_points(display, manual.points)
        elif front_to_image is not None:
            draw_grid(display, front_to_image)
            draw_points(display, selected_points)

        put_status(display, mode, image_to_front is not None, len(manual.points))
        cv2.imshow(WINDOW_CAMERA, display)

        if image_to_front is not None:
            frontal = cv2.warpPerspective(frame, image_to_front, (OUTPUT_SIZE, OUTPUT_SIZE))
            cv2.imshow(WINDOW_FRONT, frontal)
        else:
            destroy_window_if_exists(WINDOW_FRONT)

        key = cv2.waitKey(1) & 0xFF

        if mode == "view":
            if key == 27:
                break
            if key == ord("q"):
                mode = "qr"
                print("Modo QR: apunta el QR y presiona cualquier tecla para computar.")
            elif key == ord("h"):
                mode = "manual"
                manual.start()
                print("Modo manual: hace clic en los 4 vertices del cuadrado.")

        elif mode == "qr":
            # En modo QR, cualquier tecla intenta fijar la homografia y vuelve a visualizar.
            if key != 255:
                qr_points = detect_qr_points(frame, qr_detector)
                if qr_points is not None:
                    image_to_front, front_to_image, selected_points = compute_homographies(qr_points)
                    print("Homografia calculada a partir del QR.")
                else:
                    print("No se detecto QR. Se conserva la homografia anterior.")
                mode = "view"

        elif mode == "manual":
            if len(manual.points) == 4:
                image_to_front, front_to_image, selected_points = compute_homographies(manual.points)
                manual.stop()
                mode = "view"
                print("Homografia calculada a partir de 4 puntos manuales.")
            elif key != 255:
                manual.stop()
                mode = "view"
                print("Seleccion manual abortada. Se conserva la homografia anterior.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
