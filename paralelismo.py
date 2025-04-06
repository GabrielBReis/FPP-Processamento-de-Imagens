import numpy as np
import cv2
import multiprocessing as mp
from filtros import aplicar_sobel

def processar_trecho(trecho):
    return aplicar_sobel(trecho)

def aplicar_sobel_paralelo(img_gray, num_processos=None):
    if num_processos is None:
        num_processos = mp.cpu_count()

    altura = img_gray.shape[0]
    fatias = np.array_split(img_gray, num_processos, axis=0)

    with mp.Pool(processes=num_processos) as pool:
        resultados = pool.map(processar_trecho, fatias)

    resultado_final = np.vstack(resultados)
    return resultado_final
