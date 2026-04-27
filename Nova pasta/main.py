import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('./Pratica9/lena low.png')
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
plt.plot(histogram, color='black')
plt.title("Histograma da Imagem em Escala de Cinza")
plt.xlabel("Intensidade de Pixels")
plt.ylabel("Frequência da Intensidade")
plt.bar(range(256), histogram.ravel(), color='black', width=1)
plt.xlim([0, 256])
plt.show()
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('./Pratica9/vaso normal.png')
img_eq = cv2.equalizeHist(img)
img = cv2.imread("exemplo.jpg", cv2.IMREAD_GRAYSCALE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# cliplimit: contraste máximo por bin do histograma
# quanto menor, mais suave será
# tileGridSize: quantidade de divisões da imagem
# quanto maior, mais divisões e mais intenso (gerando ruído)
equalizada = clahe.apply(img)
from skimage import exposure
ajustada = exposure.match_histograms(image, reference, channel_axis=-1)
# channel_axis -1: imagens coloridas
# channel_axis None: imagem em escala de cinza