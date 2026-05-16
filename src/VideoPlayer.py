"""
VideoPlayer — odtwarzacz wideo oparty na OpenCV.
Umożliwia przetwarzanie klatek (np. detekcję ArUco) w czasie rzeczywistym.
"""
 
import cv2
import os
import platform
from typing import Callable
 
class VideoPlayer:
    """
    Odtwarza plik wideo klatka po klatce przez OpenCV.
    Każdą klatkę można przetworzyć przez podanie funkcji callback.
 
    Przykład użycia:
        def moje_przetwarzanie(klatka):
            # np. detekcja ArUco
            return klatka_z_adnotacjami
 
        p = VideoPlayer("/data/raw/GX010280.MP4")
        p.odtwarzaj(callback=moje_przetwarzanie)
    """
 
    def __init__(self, sciezka_pliku: str):
        """
        Args:
            sciezka_pliku: Ścieżka do pliku wideo.
        """
        if not os.path.exists(sciezka_pliku):
            raise FileNotFoundError(f"Plik nie istnieje: {sciezka_pliku}")
 
        self.sciezka_pliku = sciezka_pliku
        self._cap: cv2.VideoCapture | None = None
        self._info: dict | None = None
 
    # ------------------------------------------------------------------
    # Właściwości (lazy-load przy pierwszym użyciu)
    # ------------------------------------------------------------------
 
    @property
    def info(self) -> dict:
        """Metadane wideo (rozdzielczość, fps, liczba klatek, czas trwania)."""
        if self._info is None:
            cap = cv2.VideoCapture(self.sciezka_pliku)
            self._info = {
                "szerokosc":    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "wysokosc":     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps":          cap.get(cv2.CAP_PROP_FPS),
                "liczba_klatek": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "czas_trwania": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
                "kodek":        int(cap.get(cv2.CAP_PROP_FOURCC)),
            }
            cap.release()
        return self._info
 
    # ------------------------------------------------------------------
    # Metody publiczne
    # ------------------------------------------------------------------
 
    def pokaz_info(self) -> None:
        """Wyświetla metadane wideo w konsoli."""
        i = self.info
        czas = int(i["czas_trwania"])
        minuty, sekundy = divmod(czas, 60)
        godziny, minuty = divmod(minuty, 60)
        kodek_raw = i["kodek"]
        kodek = "".join([chr((kodek_raw >> (8 * j)) & 0xFF) for j in range(4)])
 
        print("\n--- Informacje o pliku ---")
        print(f"Plik         : {os.path.basename(self.sciezka_pliku)}")
        print(f"Rozdzielczość: {i['szerokosc']}x{i['wysokosc']}")
        print(f"FPS          : {i['fps']:.2f}")
        print(f"Klatki łącznie: {i['liczba_klatek']}")
        print(f"Czas trwania : {godziny:02}:{minuty:02}:{sekundy:02}")
        print(f"Kodek        : {kodek}")
        print("--------------------------\n")
 
    def odtwarzaj(
        self,
        callback: Callable[[cv2.typing.MatLike], cv2.typing.MatLike] | None = None,
        nazwa_okna: str = "VideoPlayer",
        max_szerokosc: int = 1500,
        od_klatki: int = 0,
        do_klatki: int | None = None,
    ) -> None:
        """
        Odtwarza wideo klatka po klatce w oknie OpenCV.
 
        Args:
            callback:    Funkcja (klatka) -> klatka wywoływana przed wyświetleniem.
                         Jeśli None, klatki są wyświetlane bez zmian.
            nazwa_okna:  Tytuł okna OpenCV.
            od_klatki:   Numer klatki startowej (0-based).
            do_klatki:   Numer klatki końcowej (None = do końca pliku).
 
        Sterowanie:
            SPACJA  — pauza / wznowienie
            Q / ESC — zamknięcie
            → (strzałka prawo) — +10 klatek
            ← (strzałka lewo)  — -10 klatek
        """
        cap = cv2.VideoCapture(self.sciezka_pliku)
        if not cap.isOpened():
            raise RuntimeError(f"Nie można otworzyć pliku: {self.sciezka_pliku}")
 
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        opoznienie_ms = max(1, int(1000 / fps))
        liczba_klatek = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        koniec = min(do_klatki, liczba_klatek) if do_klatki else liczba_klatek
 
        if od_klatki > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, od_klatki)
 
        pauza = False
        print(f"Odtwarzanie: {os.path.basename(self.sciezka_pliku)}")
        print("Sterowanie: SPACJA=pauza  Q/ESC=wyjście  →/←=±10 klatek")
 
        try:
            while True:
                numer_klatki = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
 
                if numer_klatki >= koniec:
                    break
 
                if not pauza:
                    ok, klatka = cap.read()
                    if not ok:
                        break
 
                    # Wywołaj przetwarzanie użytkownika
                    if callback is not None:
                        klatka = callback(klatka)
 
                    # Skalowanie do max_szerokosc (zachowuje proporcje)
                    klatka = self._skaluj(klatka, max_szerokosc)
                    # Pasek statusu na dole klatki
                    klatka = self._rysuj_status(klatka, numer_klatki, liczba_klatek, fps)
                    cv2.imshow(nazwa_okna, klatka)
 
                klawisz = cv2.waitKey(opoznienie_ms) & 0xFF
 
                if klawisz in (ord("q"), 27):   # Q lub ESC
                    break
                elif klawisz == ord(" "):        # SPACJA
                    pauza = not pauza
                elif klawisz == 83:              # → (strzałka prawo)
                    pos = min(numer_klatki + 10, koniec - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                elif klawisz == 81:              # ← (strzałka lewo)
                    pos = max(numer_klatki - 10, od_klatki)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
 
        finally:
            cap.release()
            cv2.destroyAllWindows()
 
    def iteruj_klatki(
        self,
        co_ile: int = 1,
        od_klatki: int = 0,
        do_klatki: int | None = None,
    ):
        """
        Generator zwracający kolejne klatki bez otwierania okna GUI.
        Przydatny do batch processingu (np. zapis wyników do pliku).
 
        Args:
            co_ile:    Pobierz co N-tą klatkę (domyślnie każdą).
            od_klatki: Klatka startowa.
            do_klatki: Klatka końcowa (None = do końca).
 
        Yields:
            (numer_klatki: int, klatka: np.ndarray)
 
        Przykład:
            for nr, klatka in player.iteruj_klatki(co_ile=5):
                wynik = detector.detect(klatka)
        """
        cap = cv2.VideoCapture(self.sciezka_pliku)
        if not cap.isOpened():
            raise RuntimeError(f"Nie można otworzyć pliku: {self.sciezka_pliku}")
 
        liczba_klatek = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        koniec = min(do_klatki, liczba_klatek) if do_klatki else liczba_klatek
 
        if od_klatki > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, od_klatki)
 
        try:
            numer = od_klatki
            while numer < koniec:
                ok, klatka = cap.read()
                if not ok:
                    break
                if (numer - od_klatki) % co_ile == 0:
                    yield numer, klatka
                numer += 1
        finally:
            cap.release()
 
    # ------------------------------------------------------------------
    # Metody prywatne
    # ------------------------------------------------------------------
 
    @staticmethod
    def _skaluj(klatka, max_szerokosc: int):
        """Skaluje klatkę do max_szerokosc zachowując proporcje. Nie powiększa."""
        h, w = klatka.shape[:2]
        if w <= max_szerokosc:
            return klatka
        skala = max_szerokosc / w
        nowe_wymiary = (max_szerokosc, int(h * skala))
        return cv2.resize(klatka, nowe_wymiary, interpolation=cv2.INTER_AREA)
 
    @staticmethod
    def _rysuj_status(
        klatka: cv2.typing.MatLike,
        numer: int,
        total: int,
        fps: float,
    ) -> cv2.typing.MatLike:
        """Rysuje pasek postępu i numer klatki na dole obrazu."""
        h, w = klatka.shape[:2]
        czas_s = numer / fps
        minuty, sekundy = divmod(int(czas_s), 60)
        tekst = f"Klatka {numer}/{total}  {minuty:02}:{sekundy:02}"
 
        # Tło paska
        cv2.rectangle(klatka, (0, h - 30), (w, h), (0, 0, 0), -1)
        # Pasek postępu
        postep = int(w * numer / total) if total else 0
        cv2.rectangle(klatka, (0, h - 30), (postep, h), (0, 200, 0), -1)
        # Tekst
        cv2.putText(klatka, tekst, (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return klatka
 
    def __repr__(self) -> str:
        return f"VideoPlayer(sciezka_pliku={self.sciezka_pliku!r})"