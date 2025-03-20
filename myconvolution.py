import numpy as np  # Biblioteca para manejar matrices y cálculos numéricos
import cv2  # OpenCV para manipulación de imágenes
import matplotlib.pyplot as plt  # Matplotlib para visualizar las imágenes

def convolution(image, kernel, average=False):
    """
    Aplica una convolución a una imagen en color sin convertirla a escala de grises.

    Parámetros:
    - image: Imagen en color (matriz 3D con canales RGB).
    - kernel: Matriz 2D (filtro).
    - average: Si es True, divide por el tamaño del kernel para suavizar.

    Retorna:
    - Imagen convolucionada con el mismo número de canales.
    """

    # Verifica si la imagen tiene 3 canales (RGB) o 1 canal (escala de grises)
    if len(image.shape) == 3:
        print("Imagen con 3 canales (RGB):", image.shape)
    else:
        print("Imagen con 1 canal (escala de grises):", image.shape)

    print("Tamaño del Kernel:", kernel.shape)

    # Obtiene dimensiones de la imagen y del kernel
    image_row, image_col, num_channels = image.shape  # Alto, ancho y canales (R, G, B)
    kernel_row, kernel_col = kernel.shape  # Alto y ancho del kernel

    # Calcula el padding para evitar pérdida de información en los bordes
    pad_height = kernel_row // 2  # Padding vertical
    pad_width = kernel_col // 2  # Padding horizontal

    # Crea una imagen con padding (relleno con ceros alrededor)
    padded_image = np.zeros((image_row + 2 * pad_height, image_col + 2 * pad_width, num_channels), dtype=np.float32)
    
    # Copia la imagen original dentro de la imagen con padding
    padded_image[pad_height:-pad_height, pad_width:-pad_width] = image

    # Matriz de salida con el mismo tamaño de la imagen original
    output = np.zeros_like(image, dtype=np.float32)

    # Recorre cada canal de la imagen (R, G, B) para aplicar la convolución
    for channel in range(num_channels):  # Iterar sobre los 3 canales de color
        for row in range(image_row):  # Iterar sobre las filas de la imagen
            for col in range(image_col):  # Iterar sobre las columnas de la imagen
                # Extrae la región de la imagen correspondiente al tamaño del kernel
                region = padded_image[row:row + kernel_row, col:col + kernel_col, channel]
                
                # Aplica la convolución: multiplicación elemento a elemento y suma
                output[row, col, channel] = np.sum(region * kernel)

                # Si average es True, divide por el número de elementos del kernel
                if average:
                    output[row, col, channel] /= (kernel.shape[0] * kernel.shape[1])

    # Asegura que los valores estén dentro del rango válido de píxeles (0-255)
    output = np.clip(output, 0, 255).astype(np.uint8)

    print("Imagen de salida con tamaño:", output.shape)

    return output  # Retorna la imagen convolucionada

# -------- Ejemplo de uso -------- #
# Cargar una imagen en color desde el archivo (debe estar en el mismo directorio)
image = cv2.imread("Turquia.jpg")  # Lee la imagen en formato BGR (OpenCV)

# Definir un filtro (kernel) de detección de bordes (Sobel)
sobel_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Aplicar convolución a la imagen con el kernel definido
output_image = convolution(image, sobel_kernel, average=False)

# Mostrar la imagen original y la imagen convolucionada juntas
plt.figure(figsize=(10, 5))  # Define el tamaño de la figura

# Subplot para la imagen original
plt.subplot(1, 2, 1)  # 1 fila, 2 columnas, primera imagen
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convierte de BGR a RGB para Matplotlib
plt.title("Imagen Original")  # Título de la imagen original
plt.axis("off")  # Oculta los ejes

# Subplot para la imagen convolucionada
plt.subplot(1, 2, 2)  # 1 fila, 2 columnas, segunda imagen
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))  # Convierte BGR a RGB
plt.title("Imagen Convolucionada")  # Título de la imagen procesada
plt.axis("off")  # Oculta los ejes

plt.show()  # Muestra las dos imágenes en la misma figura

# Guardar la imagen resultante en un archivo
cv2.imwrite("imagen_convolucionada.jpg", output_image)  # Guarda la imagen convolucionada

