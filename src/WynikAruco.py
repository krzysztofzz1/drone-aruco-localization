class WynikAruco:
    """Wynik detekcji ArUco dla jednej klatki."""
    def __init__(self, numer_klatki: int, markery: list):
        self.numer_klatki = numer_klatki
        self.markery = markery
 
    @property
    def liczba_markerow(self) -> int:
        return len(self.markery)
 
    def __repr__(self) -> str:
        return f"WynikAruco(klatka={self.numer_klatki}, markery={[m.id for m in self.markery]})"