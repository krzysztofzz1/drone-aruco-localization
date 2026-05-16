import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2

from VideoPlayer import VideoPlayer

# Podając ścieżkę bezpośrednio
print(os.listdir())
print(os.listdir("/data/raw"))  # powinno pokazać GX010280.MP4

sciezka = sys.argv[1] if len(sys.argv) > 1 else "/data/raw/GX010280.MP4"
player = VideoPlayer(sciezka)
player.pokaz_info()

# --- Przykładowy callback z detekcją ArUco ---
aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()
detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

def wykryj_aruco(klatka: np.ndarray) -> np.ndarray:
    szara = cv2.cvtColor(klatka, cv2.COLOR_BGR2GRAY)
    narozniki, ids, odrzucone = detector.detectMarkers(szara)
    if ids is not None:
        print(f"Wykryto markery ArUco: {ids.flatten()}")
        cv2.aruco.drawDetectedMarkers(klatka, narozniki, ids)
    return klatka

player.odtwarzaj(callback=wykryj_aruco)

xpoints = np.array([1, 8])
ypoints = np.array([3, 10])

#plt.plot(xpoints, ypoints)
#plt.show()

print("Hello world")