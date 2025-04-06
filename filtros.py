import cv2
import numpy as np

def aplicar_sobel(img_gray):
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    bordas = cv2.magnitude(sobel_x, sobel_y)
    bordas = np.uint8(np.clip(bordas, 0, 255))
    return bordas


'''
import numpy as np
import cv2

def aplicar_filtro_sobel(img_gray):
    """
    Aplica o filtro de Sobel manualmente com convolução.
    """
    # Matrizes de convolução (kernels de Sobel)
    kernel_X = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    kernel_Y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    # Aplica a convolução manualmente (cv2.filter2D faz a operação de convolução)
    sobel_x = cv2.filter2D(img_gray, cv2.CV_64F, kernel_X)
    sobel_y = cv2.filter2D(img_gray, cv2.CV_64F, kernel_Y)

    # Combina os gradientes
    bordas = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normaliza e converte para uint8
    bordas = np.clip(bordas, 0, 255)
    bordas = bordas.astype(np.uint8)

    return bordas   
'''