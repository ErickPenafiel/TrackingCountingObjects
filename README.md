# Detección y Conteo de Objetos con YOLOv8 y DeepSORT

Este proyecto implementa un sistema de detección y conteo de objetos que cruzan ciertos límites utilizando el modelo de segmentación YOLOv8-m y el modelo de trackeo DeepSORT. La visualización se realiza mediante la biblioteca OpenCV (cv2).

## Descripción

El sistema detecta y cuenta objetos que cruzan líneas definidas en el video. Para lograr esto, se utilizan los siguientes componentes:

- **YOLOv8-m**: Un modelo de segmentación que detecta objetos en cada cuadro del video.
- **DeepSORT**: Un modelo de seguimiento que asigna identificadores únicos a los objetos detectados para rastrear su movimiento a través de los cuadros.
- **OpenCV (cv2)**: Una biblioteca para la visualización y manipulación de imágenes y videos.

El proyecto está organizado en varios archivos, incluyendo un archivo `utils.py` que contiene todas las funciones necesarias para el funcionamiento del sistema.

## Requisitos

- Python 3.7+
- OpenCV
- PyTorch
- YOLOv8
- DeepSORT

Puedes instalar las dependencias necesarias ejecutando:

```bash
pip install -r requirements.txt
```

## Instalación
1. Clona este repositorio:
```bash
git clone https://github.com/tu_usuario/tu_repositorio.git
cd tu_repositorio
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Descarga el modelo YOLOv8seg-m y DeepSORT.

## Uso
1. **(Preprocesamiento del Video)**: Coloca el video en la carpeta videos/.

2. **(Configuración de los Límites)**: Define las líneas en el video que los objetos deben cruzar para ser contados. Esto se puede configurar en el archivo de configuración.

3. **(Ejecución del Sistema)**: Ejecuta el script tracking.py para comenzar la detección y el conteo:
```bash
python tracking.py
```

Si deseas cambiar el video de entrada, puedes hacerlo modificando la línea 12 del archivo tracking.py, señalando el directorio del nuevo video:

```python
cap = cv2.VideoCapture("ruta/al/video.mp4")
```

4. **(Visualización)**: Los resultados se mostrarán en tiempo real, y se guardarán en un archivo de salida si se especifica.