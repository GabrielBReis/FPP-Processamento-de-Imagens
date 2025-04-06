import cv2
import time
from filtros import aplicar_sobel
from paralelismo import aplicar_sobel_paralelo

# Carrega imagem e converte pra escala de cinza
img = cv2.imread("img01.jpg")
if img is None:
    print("Erro ao carregar a imagem.")
else:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Executa de forma sequencial
inicio_seq = time.time()
resultado_seq = aplicar_sobel(img_gray)
fim_seq = time.time()

# Executa de forma paralela
inicio_par = time.time()
resultado_par = aplicar_sobel_paralelo(img_gray)
fim_par = time.time()

# Exibe tempos
print(f"Tempo sequencial: {fim_seq - inicio_seq:.4f} s")
print(f"Tempo paralelo:   {fim_par - inicio_par:.4f} s")

# Mostra resultados
from matplotlib import pyplot as plt

plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.title("Sequencial")
plt.imshow(resultado_seq, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Paralelo")
plt.imshow(resultado_par, cmap='gray')

plt.show()
