import numpy as np

class Marker:
    """Pojedynczy wykryty marker ArUco."""
    def __init__(self, id: int, narozniki: np.ndarray,
                 rvec: np.ndarray | None = None, tvec: np.ndarray | None = None):
        self.id = id
        self.narozniki = narozniki
        self.rvec = rvec
        self.tvec = tvec
 
    @property
    def srodek(self) -> tuple[int, int]:
        """Środek markera w pikselach."""
        cx = int(np.mean(self.narozniki[:, 0]))
        cy = int(np.mean(self.narozniki[:, 1]))
        return cx, cy