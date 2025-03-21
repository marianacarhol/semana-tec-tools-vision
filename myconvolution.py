# Mariana Carrillo Holguin
# A01253358
# MANUAL
'''
Para ejecutar el programa correctamente se debe:
- Descargar previamente en la computadora el lenguaje Python y las librerías Numpy, CV2 y Matplotlib
- En caso de que el programa no cargue la imagen "Turquia.jpg" correctamente, verifique que la ruta (path) de la imagen sea la adecuada. 
'''

import numpy as np  # Biblioteca para manejar matrices y cálculos numéricos
import cv2  # OpenCV para manipulación de imágenes
import matplotlib.pyplot as plt  # Matplotlib para visualizar las imágenes

def convolution(image, kernel, average=False):    
    # Obtiene dimensiones de la imagen y del kernel
    image_row, image_col, num_channels = image.shape  # Alto, ancho y canales (R, G, B)
    kernel_row, kernel_col = kernel.shape  # Alto y ancho del kernel
    
    # Calcula las dimensiones de la imagen resultante, considerando que no se usará padding
    output_row = image_row - kernel_row + 1  # Alto de la imagen de salida
    output_col = image_col - kernel_col + 1  # Ancho de la imagen de salida
    
    # Inicializa la matriz de salida con ceros, del tamaño adecuado y el mismo número de canales
    output = np.zeros((output_row, output_col, num_channels), dtype=np.float32)

    # Recorre cada canal de la imagen (R, G, B) para aplicar la convolución
    for channel in range(num_channels):  # Itera sobre los canales de color (0: R, 1: G, 2: B)
        for row in range(output_row):  # Recorre las filas válidas sin exceder los bordes
            for col in range(output_col):  # Recorre las columnas válidas sin exceder los bordes
                # Extrae una región de la imagen del mismo tamaño que el kernel
                region = image[row:row + kernel_row, col:col + kernel_col, channel]
                
                # Realiza la multiplicación elemento a elemento entre la región y el kernel, y suma el resultado
                output[row, col, channel] = np.sum(region * kernel)

                # Si average es True, divide por el número de elementos del kernel para suavizar
                if average:
                    output[row, col, channel] /= (kernel.shape[0] * kernel.shape[1])

    # Limita los valores de la matriz resultante al rango de píxeles [0, 255] y convierte a enteros sin signo (uint8)
    output = np.clip(output, 0, 255).astype(np.uint8)

    return output  # Retorna la imagen convolucionada sin padding

# Carga una imagen en color desde el archivo especificado (debe estar en el mismo directorio que el script)
image = cv2.imread("Turquia.jpg")  # Lee la imagen en formato BGR que utiliza OpenCV por defecto

# Define un kernel de detección de bordes (Sobel horizontal en este caso)
sobel_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Aplica la convolución a la imagen usando el kernel de Sobel
output_image = convolution(image, sobel_kernel, average=False)

# Visualiza la imagen original y la imagen resultante tras la convolución
plt.figure(figsize=(10, 5))  # Establece el tamaño del lienzo para mostrar las imágenes

# Muestra la imagen original
plt.subplot(1, 2, 1)  # Crea el primer subplot en una figura de 1 fila y 2 columnas
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convierte la imagen de BGR a RGB para mostrarla correctamente
plt.title("Imagen Original")  # Asigna un título descriptivo
plt.axis("off")  # Oculta los ejes para una presentación más limpia

# Muestra la imagen convolucionada
plt.subplot(1, 2, 2)  # Crea el segundo subplot
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))  # Convierte y muestra la imagen resultante
plt.title("Imagen Convolucionada")  # Asigna un título descriptivo
plt.axis("off")  # Oculta los ejes

plt.show()  # Renderiza y muestra ambas imágenes en pantalla