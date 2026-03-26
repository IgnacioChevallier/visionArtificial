import cv2
import numpy as np
import os


def perspective_tilt(img, output_folder, count):
    h, w = img.shape

    tilts = [0.2, 0.3]  # intensidad

    for t in tilts:
        dx = int(w * t)

        pts1 = np.float32([
            [0, 0],
            [w, 0],
            [0, h],
            [w, h]
        ])

        # inclinación hacia atrás
        pts_back = np.float32([
            [dx, 0],
            [w - dx, 0],
            [0, h],
            [w, h]
        ])

        M_back = cv2.getPerspectiveTransform(pts1, pts_back)

        warped_back = cv2.warpPerspective(
            img, M_back, (w, h),
            borderValue=255
        )

        cv2.imwrite(os.path.join(output_folder, f"img_{count}.png"), warped_back)
        count += 1

        # inclinación hacia adelante
        pts_front = np.float32([
            [0, 0],
            [w, 0],
            [dx, h],
            [w - dx, h]
        ])

        M_front = cv2.getPerspectiveTransform(pts1, pts_front)

        warped_front = cv2.warpPerspective(
            img, M_front, (w, h),
            borderValue=255
        )

        cv2.imwrite(os.path.join(output_folder, f"img_{count}.png"), warped_front)
        count += 1

    return count


def augment_image(image_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("No se pudo cargar la imagen")
        return

    # BINARIZAR (MUY IMPORTANTE)
    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    h, w = img.shape
    count = 0

    #  ROTACIONES
    for angle in range(0, 360, 15):
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=255)

        cv2.imwrite(os.path.join(output_folder, f"img_{count}.png"), rotated)
        count += 1

    #  ESCALAS
    scales = [0.5, 0.75, 1.25, 1.5]

    for scale in scales:
        resized = cv2.resize(img, None, fx=scale, fy=scale)

        canvas = np.ones((h, w), dtype=np.uint8) * 255

        rh, rw = resized.shape
        y_offset = (h - rh) // 2
        x_offset = (w - rw) // 2

        if y_offset >= 0 and x_offset >= 0:
            canvas[y_offset:y_offset+rh, x_offset:x_offset+rw] = resized
        else:
            resized = cv2.resize(resized, (w, h))
            canvas = resized

        cv2.imwrite(os.path.join(output_folder, f"img_{count}.png"), canvas)
        count += 1

    #  PERSPECTIVA (adelante + atrás)
    count = perspective_tilt(img, output_folder, count)

    print(f"Se generaron {count} imágenes")


# 👉 USO
augment_image("utils/diamond.png", "../dataset/diamond")