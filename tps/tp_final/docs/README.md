# TP Final - Dataset satelital de areas quemadas

Este proyecto genera un dataset de areas quemadas para la zona del Parque Nacional Los Alerces a partir de imagenes Sentinel-2. El flujo principal compara una composicion previa al incendio con otra posterior, calcula una mascara binaria de zonas quemadas y exporta tiles para su inspeccion o uso posterior.

## Objetivo

Construir un conjunto de datos con:

- imagenes RGB previas al incendio;
- imagenes RGB posteriores al incendio;
- mascaras binarias de areas quemadas;
- metadatos por tile, como coordenadas, fechas y porcentaje quemado.

La mascara se obtiene sin aprendizaje automatico ni etiquetas externas: se deriva directamente del indice `dNBR` calculado entre las composiciones `PRE` y `POST`.

## Estructura del proyecto

```text
tp_final/
├── generar_dataset.py        # descarga imagenes y genera mascaras
├── visualizar.py             # explora las imagenes y sus mascaras
├── requirements.txt          # dependencias del proyecto
├── docs/
│   └── README.md             # descripcion actual del proyecto
└── dataset/
    ├── imagenes_pre/          # tiles PRE incendio, multibanda .tif
    ├── imagenes/              # tiles POST incendio, multibanda .tif
    ├── mascaras/              # mascaras binarias .tif
    └── metadata.csv           # informacion tabular por tile
```

Cada tile exportado contiene las bandas Sentinel-2 `B2`, `B3`, `B4`, `B8`, `B11` y `B12`. Para visualizar color verdadero se usan `B4`, `B3` y `B2` como canales rojo, verde y azul.

## Como se generan las mascaras

El script `generar_dataset.py` usa la coleccion `COPERNICUS/S2_SR_HARMONIZED` de Google Earth Engine sobre una region rectangular que cubre Los Alerces.

### 1. Seleccion temporal

Se construyen dos ventanas:


| Composicion | Fechas                      | Motivo                                  |
| ----------- | --------------------------- | --------------------------------------- |
| `PRE`       | `2025-11-01` a `2025-12-08` | periodo anterior al inicio del incendio |
| `POST`      | `2026-02-05` a `2026-02-20` | periodo con la cicatriz ya formada      |


### 2. Filtrado de escenas y nubes

Primero se filtran escenas con menos de `20%` de nubosidad. Luego se define una mascara por la banda `SCL` para excluir:

- sombras de nube (`3`);
- nubes de probabilidad media (`8`);
- nubes de probabilidad alta (`9`);
- cirrus (`10`);
- nieve (`11`).

El proyecto conserva dos versiones de las composiciones:

- `pre_masked` y `post_masked`, usadas para calcular la mascara de incendio con menos contaminacion atmosférica;
- `pre_full` y `post_full`, usadas para exportar imagenes RGB completas y evitar huecos negros en la visualizacion.

### 3. Calculo de NBR y dNBR

Para cada composicion enmascarada se calcula el indice `NBR`:

```text
NBR = (B8 - B12) / (B8 + B12)
```

Luego se calcula la diferencia entre el estado previo y posterior:

```text
dNBR = NBR_pre - NBR_post
```

Un valor alto de `dNBR` indica una caida marcada de vegetacion sana y aumento de suelo/ceniza expuesta, patron compatible con area quemada.

### 4. Binarizacion y limpieza

La mascara inicial se obtiene con:

```text
quemado = dNBR > 0.25
```

Despues se eliminan componentes conexas menores a `100` pixeles mediante `connectedPixelCount`. Esto reduce manchas pequenas y falsos positivos aislados.

El resultado final es una mascara binaria:

- `1`: quemado;
- `0`: no quemado.

### 5. Exportacion por tiles

La region se divide en tiles de aproximadamente `256 x 256` pixeles a `20 m/pixel`. Para cada tile se exportan:

- imagen `PRE` en `dataset/imagenes_pre/`;
- imagen `POST` en `dataset/imagenes/`;
- mascara binaria en `dataset/mascaras/`;
- una fila en `metadata.csv` con coordenadas, fechas y porcentaje quemado.

## Visualizacion

El script `visualizar.py` permite recorrer los tiles con las flechas del teclado y muestra cuatro paneles:

1. imagen antes del incendio;
2. imagen despues del incendio;
3. mascara binaria en blanco y negro;
4. superposicion de la mascara sobre la imagen posterior.

En la mascara:

- negro representa `No quemado (0)`;
- blanco representa `Quemado (1)`.

Para mejorar la lectura visual, las imagenes RGB se normalizan por percentiles `2-98` y se aplica una correccion gamma suave (`GAMMA = 0.8`). Esa mejora afecta solo a la visualizacion, no modifica los valores almacenados en los `.tif`.

## Uso basico

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Autenticar Earth Engine una vez:

```bash
earthengine authenticate
earthengine authenticate --auth_mode=notebook --force
```

Generar el dataset:

```bash
python generar_dataset.py
```

Explorar los tiles:

```bash
python visualizar.py
```

Abrir desde un tile especifico:

```bash
python visualizar.py tile_000
```

Durante la visualizacion:

- flecha izquierda/derecha: navegar entre tiles;
- `s`: guardar la figura actual como PNG;
- `q`: cerrar la ventana.

## Parametros principales


| Variable      | Valor actual | Funcion                                 |
| ------------- | ------------ | --------------------------------------- |
| `DNBR_UMBRAL` | `0.25`       | umbral para clasificar area quemada     |
| `MIN_PIXELES` | `100`        | tamano minimo de una region conectada   |
| `ESCALA`      | `20`         | resolucion espacial en metros por pixel |
| `N_TILES`     | `30`         | cantidad maxima de tiles a exportar     |


