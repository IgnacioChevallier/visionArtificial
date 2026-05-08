import os
import sys

# OpenCV 4.10+ prueba el backend Orbbec (OBSENSOR) antes que V4L2 y ensucia el log / el orden
# de backends; prioridad 0 lo desactiva salvo que lo definas vos (debe ir antes de import cv2).
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_OBSENSOR", "0")

import numpy as np
import cv2


def destroy_window_if_exists(name):
    try:
        cv2.destroyWindow(name)
    except cv2.error:
        pass


def _camera_index():
    """Índice fijo si se definió HOMOGRAFIA_CAMERA u OPENCV_CAMERA_INDEX."""
    for key in ("HOMOGRAFIA_CAMERA", "OPENCV_CAMERA_INDEX"):
        v = os.environ.get(key)
        if v is not None:
            return int(v)
    return None


def _capture_delivers_frame(cap, attempts=45):
    """isOpened() a veces es true sin stream; exigimos al menos un frame válido."""
    for _ in range(attempts):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return True
    return False


def _try_open_capture_index(index):
    """Prueba /dev/videoN + V4L2 y luego índice con backend por defecto."""
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


def open_video_capture(index=None):
    """
    Fuente por prioridad:
    1. HOMOGRAFIA_VIDEO (ruta a archivo o dispositivo, ej. /dev/video2)
    2. HOMOGRAFIA_CAMERA / OPENCV_CAMERA_INDEX → solo ese índice
    3. Sin eso → prueba índices 0..HOMOGRAFIA_SCAN_LAST (default 8)
    """
    path = (os.environ.get("HOMOGRAFIA_VIDEO") or "").strip()
    if path:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return None
        if not _capture_delivers_frame(cap):
            cap.release()
            return None
        return cap

    fixed = _camera_index()
    if index is not None:
        indices = [index]
    elif fixed is not None:
        indices = [fixed]
    else:
        last = max(0, int(os.environ.get("HOMOGRAFIA_SCAN_LAST", "4")))
        indices = list(range(0, last + 1))

    scanned = len(indices) > 1
    for idx in indices:
        cap = _try_open_capture_index(idx)
        if cap is not None:
            if scanned:
                print(f"Cámara: índice {idx}", file=sys.stderr)
            return cap
    return None


def _camera_open_fail_message():
    path = (os.environ.get("HOMOGRAFIA_VIDEO") or "").strip()
    if path:
        return (
            f"No se pudo abrir HOMOGRAFIA_VIDEO={path!r}.\n"
            "  • Comprobá que el archivo existe o que el device es capturable.\n"
        )
    return (
        "No se pudo abrir ninguna cámara (se probaron los índices configurados).\n"
        f"  • Si sabés el índice: HOMOGRAFIA_CAMERA=2 python homografia.py\n"
        "  • O un device concreto: HOMOGRAFIA_VIDEO=/dev/video2 python homografia.py\n"
        "  • O archivo de vídeo: HOMOGRAFIA_VIDEO=/ruta/al/archivo.mp4\n"
        "  • ¿Seguís sin imagen con permisos bien? Probá HOMOGRAFIA_VIDEO=/dev/video0\n"
        f"  (barrido por defecto hasta HOMOGRAFIA_SCAN_LAST; sin definir vale 4).\n"
    )


def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    if maxWidth < 2 or maxHeight < 2:
        raise ValueError("Los 4 puntos seleccionados no forman un rectangulo valido.")
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


points = []
# points = [(0, 240), (50, 150), (590, 150), (640, 240)]

#
# def main():
#     cv2.namedWindow('frame')
#     cv2.setMouseCallback('frame', on_click)
#     frame = cv2.imread('../static/images/tenis.jpg')
#     cv2.imshow('frame', frame)
#     global points
#     while len(points) <= 4:
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             if len(points) == 4:
#                 pts = np.array(points, dtype="float32")
#                 cv2.imshow('imagen', four_point_transform(frame, pts))
#                 cv2.waitKey(0)
#                 break
#     cv2.waitKey(0)
#

def calibrate_camera(cap):
    CHECKBOARD = (4, 7)

    objp = np.zeros((CHECKBOARD[0] * CHECKBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKBOARD[0], 0:CHECKBOARD[1]].T.reshape(-1, 2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    image_points = []
    object_points = []
    camera_matrix = None
    distortion_coeff = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            cv2.waitKey(30)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_gray = cv2.flip(gray, 1)
        cv2.imshow('binary', display_gray)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            ret2, corners = cv2.findChessboardCorners(gray, (CHECKBOARD[0], CHECKBOARD[1]))
            if ret2:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                image_points.append(corners2)
                object_points.append(objp)
                ret, camera_matrix, distortion_coeff, rotationvecs, translationvecs = \
                    cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
                cv2.imshow('drawedCorners',
                           cv2.drawChessboardCorners(frame, (CHECKBOARD[0], CHECKBOARD[1]), corners2, ret2))
                print("--------------CAMERA MATRIX------------")
                print(camera_matrix)
                print("--------------DISTORTION COEFF------------")
                print(distortion_coeff)
                print("Calibracion lista. Presiona 'w' para continuar con la homografia.")
                cv2.waitKey(500)
            else:
                print("No se detecto el tablero. Alinealo completo en la imagen y presiona 'q' otra vez.")
        elif key == ord('w'):
            if camera_matrix is None:
                print("Primero captura un tablero valido con 'q'.")
                continue
            (h, w, d) = frame.shape
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w, h), 1, (w, h))
            destroy_window_if_exists('binary')
            destroy_window_if_exists('drawedCorners')
            return camera_matrix, distortion_coeff, new_camera_matrix


def mainVideo():
    cap = open_video_capture()
    if cap is None:
        sys.stderr.write(_camera_open_fail_message())
        sys.exit(1)
    # cap = cv2.VideoCapture('../static/videos/carsRt9_3.avi')
    camera_matrix = None
    distortion_coeff = None
    new_camera_matrix = None
    cv2.namedWindow('imagen normal')
    cv2.namedWindow('imagen distorsioned_frame')
    cv2.setMouseCallback('imagen distorsioned_frame', on_click)
    while True:
        while camera_matrix is None:
            camera_matrix, distortion_coeff, new_camera_matrix = calibrate_camera(cap)
            for _ in range(10):
                cap.read()
        global points
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            cv2.waitKey(30)
            continue
        cv2.imshow('imagen normal', frame)
        distorsioned_frame = cv2.undistort(frame, camera_matrix, distortion_coeff, None, new_camera_matrix)
        cv2.imshow('imagen distorsioned_frame', distorsioned_frame)
        if len(points) == 4:
            pts = np.array(points, dtype="float32")
            try:
                cv2.imshow('imagen transformada', four_point_transform(distorsioned_frame, pts))
            except ValueError as exc:
                print(exc)
                points = []
        elif len(points) > 4:
            print("Se seleccionaron mas de 4 puntos. Reiniciando seleccion.")
            points = []
        if cv2.waitKey(1) & 0xFF == ord('w'):
            break


def on_click(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global points
        if len(points) >= 4:
            points = []
        points.append((x, y))
        print(f"Punto {len(points)}/4: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        points = []
        print("Seleccion reiniciada.")


mainVideo()
