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

def convolution(image, kernel, average=False, extra_padding=2):    
    # Obtiene dimensiones de la imagen y del kernel
    image_row, image_col, num_channels = image.shape  # Alto, ancho y canales (R, G, B)
    kernel_row, kernel_col = kernel.shape  # Alto y ancho del kernel
    
    # Calcula el padding necesario (sumando extra_padding para hacerlo más grueso)
    pad_height = (kernel_row // 2) + extra_padding  # Padding vertical
    pad_width = (kernel_col // 2) + extra_padding  # Padding horizontal
    
    # Crea una imagen con padding de ceros
    padded_image = np.zeros((image_row + 2 * pad_height, image_col + 2 * pad_width, num_channels), dtype=np.float32)
    
    # Copia la imagen original dentro de la imagen con padding
    padded_image[pad_height:pad_height + image_row, pad_width:pad_width + image_col] = image
    
    # Inicializa la matriz de salida con el mismo tamaño que la imagen con padding
    output = np.zeros((image_row + 2 * pad_height, image_col + 2 * pad_width, num_channels), dtype=np.float32)

    # Recorre cada canal de la imagen (R, G, B) para aplicar la convolución
    for channel in range(num_channels):  # Itera sobre los canales de color (0: R, 1: G, 2: B)
        for row in range(image_row):  # Recorre las filas incluyendo el padding
            for col in range(image_col):  # Recorre las columnas incluyendo el padding
                # Extrae una región de la imagen con el tamaño del kernel
                region = padded_image[row:row + kernel_row, col:col + kernel_col, channel]
                
                # Realiza la multiplicación elemento a elemento entre la región y el kernel, y suma el resultado
                output[row + pad_height, col + pad_width, channel] = np.sum(region * kernel)

                # Si average es True, divide por el número de elementos del kernel para suavizar
                if average:
                    output[row + pad_height, col + pad_width, channel] /= (kernel.shape[0] * kernel.shape[1])

    # Limita los valores de la matriz resultante al rango de píxeles [0, 255] y convierte a enteros sin signo (uint8)
    output = np.clip(output, 0, 255).astype(np.uint8)

    return padded_image.astype(np.uint8), output  # Retorna la imagen con padding y la imagen convolucionada

# Carga una imagen en color desde el archivo especificado (debe estar en el mismo directorio que el script)
image = cv2.imread("Turquia.jpg")  # Lee la imagen en formato BGR que utiliza OpenCV por defecto

# Define un kernel de detección de bordes (Sobel horizontal en este caso)
sobel_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Aplica la convolución a la imagen usando el kernel de Sobel con un padding más grueso
padded_image, output_image = convolution(image, sobel_kernel, average=False, extra_padding=5)

# Visualiza la imagen original, la imagen con padding y la imagen resultante tras la convolución
plt.figure(figsize=(15, 5))  # Establece el tamaño del lienzo para mostrar las imágenes

# Muestra la imagen original
plt.subplot(1, 3, 1)  # Primer subplot de 3
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Imagen Original")
plt.axis("off")

# Muestra la imagen con padding
plt.subplot(1, 3, 2)  # Segundo subplot
plt.imshow(cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB))
plt.title("Imagen con Padding")
plt.axis("off")

# Muestra la imagen convolucionada
plt.subplot(1, 3, 3)  # Tercer subplot
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title("Imagen Convolucionada con Padding")
plt.axis("off")

plt.show()  # Muestra las imágenes