import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2

from VideoPlayer import VideoPlayer
from ArUco import ArUco

# Podając ścieżkę bezpośrednio
print(os.listdir())
print(os.listdir("/data/raw"))  # powinno pokazać GX010280.MP4

sciezka = sys.argv[1] if len(sys.argv) > 1 else "/data/raw/GX010280.MP4"
player = VideoPlayer(sciezka)
player.pokaz_info()

# Bez kalibracji — tylko detekcja i rysowanie ramek
detector = ArUco(slownik=cv2.aruco.DICT_6X6_250)
player.odtwarzaj(callback=detector.callback)

# Zapis wyników do CSV
detector.eksportuj_do_csv("/data/raw/aruco_wyniki.csv")

xpoints = np.array([1, 8])
ypoints = np.array([3, 10])

#plt.plot(xpoints, ypoints)
#plt.show()

print("Hello world")