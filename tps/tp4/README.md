# Introducción

Este documento describe un proyecto introductorio al uso de la homografía como transformación de perspectiva. El proyecto requiere obtener una homografía por varios métodos, y generar algunas imágenes que demuestran que la perspectiva fue correctamente relevada.

# Proyecto

Consta de dos partes:

- Determinación de la matrix homográfica
- Visualización de la transformación de perspectiva

La determinación de la matriz homográfica se lleva a cabo a partir de la correspondencia de los vértices de un cuadrado: entre 4 keypoints identificados en la imagen de la webcam y las 4 coordenadas conocidas correspondientes a una vista frontal.

Determinada la homografía, se procede con su visualización.

# Homografía

Esta determinación se realiza con dos métodos:

## Homografía de un código QR

El método detecta las coordenadas de los 4 vértices del código QR en la imagen, con los que se puede computar la homografía.

## Homografía a partir de 4 keypoints anotados a mano

El usuario hace clic en los 4 vértices de un cuadrado visto en perspectiva.

Los métodos difieren en la manera en que se detectan los 4 puntos del cuadrado en la imagen. Uno detecta las esquinas de un código QR, el otro requiere que el usuario marque los puntos con el mouse.

# Visualización

Con la matriz obtenida se demostrará la transformación de perspectiva de dos maneras:

## Anotación de grilla

Sobre la imagen original se dibuja una grilla de celdas cuadradas en perspectiva, por ejemplo de 3x3 celdas.

## Perspectiva homográfica

En una ventana aparte se muestra una visualización frontal, del cuadrado en perspectiva.

# Interfaz

Las teclas controlan el modo de operación:

- `q`: el sistema entra al modo de detección de QR.
  - Al pulsar cualquier tecla se computa la homografía y se vuelve al modo de visualización.
  - Si no se registró ningún QR o no se puede computar la homografía, se vuelve al modo de visualización manteniendo la homografía anterior.
- `h`: el sistema entra al modo de homografía asistida, en la que el usuario debe hacer clic en 4 vértices de un cuadrado en perspectiva.
  - Luego del 4º clic se computa la homografía y se vuelve a la visualización.
  - Con cualquier tecla se aborta y se vuelve a la visualización con la homografía anterior.
