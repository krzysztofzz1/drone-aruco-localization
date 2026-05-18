import cv2
import numpy as np

from Marker import Marker
from WynikAruco import WynikAruco

class ArUco:
    """
    Detektor markerów ArUco.
 
    Przykład użycia samodzielnego:
        detector = ArUco()
        for nr, klatka in player.iteruj_klatki():
            wynik = detector.wykryj(klatka)
            print(wynik)
 
    Przykład jako callback do VideoPlayer:
        detector = ArUco(macierz_kamery=K, wspolczynniki_dystorsji=D)
        player.odtwarzaj(callback=detector.callback)
        print(detector.historia)   # lista WynikAruco dla każdej klatki
 
    Przykład z kalibracją kamery i estymacją pozy:
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        D = np.zeros((4, 1))
        detector = ArUco(
            slownik=cv2.aruco.DICT_6X6_250,
            macierz_kamery=K,
            wspolczynniki_dystorsji=D,
            dlugosc_boku_m=0.17,   # rozmiar markera w metrach
        )
    """
 
    def __init__(
        self,
        slownik: int = cv2.aruco.DICT_6X6_250,
        macierz_kamery: np.ndarray | None = None,
        wspolczynniki_dystorsji: np.ndarray | None = None,
        dlugosc_boku_m: float = 0.17,
        rysuj_osie: bool = True,
        rysuj_id: bool = True,
        kolor_ramki: tuple[int, int, int] = (0, 255, 0),
    ):
        """
        Args:
            slownik:                   Stały cv2.aruco.DICT_* określający rodzaj markerów.
            macierz_kamery:            Macierz K (3x3). Wymagana do estymacji pozy.
            wspolczynniki_dystorsji:   Współczynniki dystorsji. Wymagane do estymacji pozy.
            dlugosc_boku_m:            Długość boku markera w metrach (do estymacji pozy).
            rysuj_osie:                Rysuj osie XYZ na markerze (wymaga macierzy kamery).
            rysuj_id:                  Rysuj ID markera nad ramką.
            kolor_ramki:               Kolor ramki markera w BGR.
        """
        self.macierz_kamery = macierz_kamery
        self.wspolczynniki_dystorsji = wspolczynniki_dystorsji
        self.dlugosc_boku_m = dlugosc_boku_m
        self.rysuj_osie = rysuj_osie
        self.rysuj_id = rysuj_id
        self.kolor_ramki = kolor_ramki
 
        self._slownik = cv2.aruco.getPredefinedDictionary(slownik)
        self._parametry = cv2.aruco.DetectorParameters()
        self._detektor = cv2.aruco.ArucoDetector(self._slownik, self._parametry)
 
        self._numer_klatki: int = 0
        self.historia: list[WynikAruco] = []
 
    # ------------------------------------------------------------------
    # Metody publiczne
    # ------------------------------------------------------------------
 
    def wykryj(self, klatka: np.ndarray) -> WynikAruco:
        """
        Wykrywa markery ArUco na klatce. Nie modyfikuje obrazu.
 
        Args:
            klatka: Obraz BGR (np.ndarray).
 
        Returns:
            WynikAruco z listą wykrytych markerów.
        """
        szara = cv2.cvtColor(klatka, cv2.COLOR_BGR2GRAY)
        narozniki_raw, ids_raw, _ = self._detektor.detectMarkers(szara)
 
        markery: list[Marker] = []
        if ids_raw is not None:
            for narozniki, marker_id in zip(narozniki_raw, ids_raw.flatten()):
                m = Marker(id=int(marker_id), narozniki=narozniki[0])
 
                if self._ma_kalibracje():
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        narozniki,
                        self.dlugosc_boku_m,
                        self.macierz_kamery,
                        self.wspolczynniki_dystorsji,
                    )
                    m.rvec = rvec[0]
                    m.tvec = tvec[0]
 
                markery.append(m)
 
        wynik = WynikAruco(numer_klatki=self._numer_klatki, markery=markery)
        self.historia.append(wynik)
        self._numer_klatki += 1
        return wynik
 
    def adnotuj(self, klatka: np.ndarray, wynik: WynikAruco) -> np.ndarray:
        """
        Rysuje wykryte markery na klatce (ramki, ID, osie).
 
        Args:
            klatka: Obraz BGR do modyfikacji (kopia).
            wynik:  Wynik zwrócony przez wykryj().
 
        Returns:
            Klatka z naniesionymi adnotacjami.
        """
        klatka = klatka.copy()
 
        for marker in wynik.markery:
            pts = marker.narozniki.astype(int)
 
            # Ramka markera
            cv2.polylines(klatka, [pts], isClosed=True, color=self.kolor_ramki, thickness=2)
 
            # ID markera
            if self.rysuj_id:
                cx, cy = marker.srodek
                cv2.putText(
                    klatka, f"ID:{marker.id}",
                    (cx - 20, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )
 
            # Osie XYZ (tylko gdy dostępna kalibracja i estymacja pozy)
            if self.rysuj_osie and marker.rvec is not None and self._ma_kalibracje():
                cv2.drawFrameAxes(
                    klatka,
                    self.macierz_kamery,
                    self.wspolczynniki_dystorsji,
                    marker.rvec,
                    marker.tvec,
                    self.dlugosc_boku_m * 0.5,
                )
 
        # Licznik wykrytych markerów
        cv2.putText(
            klatka,
            f"ArUco: {wynik.liczba_markerow} markerow",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2,
        )
 
        return klatka
 
    def callback(self, klatka: np.ndarray) -> np.ndarray:
        """
        Gotowy callback do podania bezpośrednio do VideoPlayer.odtwarzaj().
        Wykrywa markery i rysuje adnotacje w jednym kroku.
 
        Użycie:
            player.odtwarzaj(callback=detector.callback)
        """
        wynik = self.wykryj(klatka)
        return self.adnotuj(klatka, wynik)
 
    def wyczysc_historie(self) -> None:
        """Czyści zapisaną historię wyników i resetuje licznik klatek."""
        self.historia.clear()
        self._numer_klatki = 0
 
    def eksportuj_do_csv(self, sciezka: str) -> None:
        """
        Zapisuje historię detekcji do pliku CSV.
 
        Kolumny: numer_klatki, marker_id, x0,y0, x1,y1, x2,y2, x3,y3,
                 srodek_x, srodek_y, tvec_x, tvec_y, tvec_z
        """
        import csv
        with open(sciezka, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "numer_klatki", "marker_id",
                "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3",
                "srodek_x", "srodek_y",
                "tvec_x", "tvec_y", "tvec_z",
            ])
            for wynik in self.historia:
                for m in wynik.markery:
                    pts = m.narozniki.flatten().tolist()
                    tvec = m.tvec.flatten().tolist() if m.tvec is not None else ["", "", ""]
                    writer.writerow([
                        wynik.numer_klatki, m.id,
                        *pts,
                        *m.srodek,
                        *tvec,
                    ])
        print(f"Zapisano {sum(w.liczba_markerow for w in self.historia)} detekcji → {sciezka}")
 
    # ------------------------------------------------------------------
    # Metody prywatne
    # ------------------------------------------------------------------
 
    def _ma_kalibracje(self) -> bool:
        return self.macierz_kamery is not None and self.wspolczynniki_dystorsji is not None
 
    def __repr__(self) -> str:
        return (
            f"ArUco(slownik={self._slownik}, kalibracja={self._ma_kalibracje()}, "
            f"klatek_przetworzonych={self._numer_klatki})"
        )