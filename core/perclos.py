# core/perclos.py
# Tanggung jawab: logika temporal PERCLOS dan keputusan kantuk akhir.
#
# PERCLOS (Percentage of Eye Closure) adalah standar NHTSA untuk
# mengukur kantuk. Berbeda dari prediksi per-frame (yang bisa
# false positive), PERCLOS mengamati tren selama N frame terakhir.

from collections import deque
from dataclasses import dataclass, field
from config import (
    PERCLOS_WINDOW, PERCLOS_THRESHOLD,
    YAWN_THRESHOLD,
)


@dataclass
class PerclosState:
    """State snapshot PERCLOS pada satu frame."""
    perclos_ratio  : float   # 0.0 – 1.0, rasio frame mata tertutup
    yawn_count     : int     # jumlah frame menguap dalam window
    is_drowsy      : bool    # keputusan akhir
    drowsy_reason  : str     # alasan: "perclos" / "yawn" / "both" / ""
    window_filled  : bool    # True jika window sudah penuh (30 frame)
    frames_in_window: int    # jumlah frame dalam window saat ini


class PerclosDetector:
    """
    Mengakumulasi status mata dan mulut per frame,
    lalu memutuskan kantuk berdasarkan tren temporal.

    Mengapa perlu temporal logic:
    - Prediksi CNN per-frame bisa salah (noise, blur sesaat)
    - Satu frame mata tertutup bukan kantuk — bisa berkedip
    - PERCLOS mengharuskan mata tertutup KONSISTEN dalam window
    """

    def __init__(
        self,
        window    : int   = PERCLOS_WINDOW,
        perclos_th: float = PERCLOS_THRESHOLD,
        yawn_th   : int   = YAWN_THRESHOLD,
    ):
        self.window     = window
        self.perclos_th = perclos_th
        self.yawn_th    = yawn_th

        # Buffer circular — otomatis buang frame terlama
        self._eye_buf   = deque(maxlen=window)
        self._mouth_buf = deque(maxlen=window)

    def update(
        self,
        left_eye : str,
        right_eye: str,
        mouth    : str,
    ) -> PerclosState:
        """
        Update buffer dengan prediksi frame saat ini.
        Kedua mata harus sama-sama tertutup untuk dihitung.

        Args:
            left_eye : "open_eye" | "closed_eye" | "yawn"
            right_eye: "open_eye" | "closed_eye" | "yawn"
            mouth    : "open_eye" | "closed_eye" | "yawn"

        Returns PerclosState dengan keputusan kantuk.
        """
        both_closed = (
            left_eye  == "closed_eye" and
            right_eye == "closed_eye"
        )
        is_yawning = (mouth == "yawn")

        self._eye_buf.append(1 if both_closed else 0)
        self._mouth_buf.append(1 if is_yawning else 0)

        n            = len(self._eye_buf)
        perclos      = sum(self._eye_buf) / n if n > 0 else 0.0
        yawn_count   = sum(self._mouth_buf)
        window_filled = n >= self.window

        # Keputusan kantuk
        by_perclos = perclos   >= self.perclos_th
        by_yawn    = yawn_count >= self.yawn_th
        is_drowsy  = by_perclos or by_yawn

        if by_perclos and by_yawn:
            reason = "perclos + menguap"
        elif by_perclos:
            reason = f"perclos {perclos*100:.0f}%"
        elif by_yawn:
            reason = f"menguap {yawn_count}x"
        else:
            reason = ""

        return PerclosState(
            perclos_ratio   = perclos,
            yawn_count      = yawn_count,
            is_drowsy       = is_drowsy,
            drowsy_reason   = reason,
            window_filled   = window_filled,
            frames_in_window= n,
        )

    def reset(self):
        """Reset buffer — panggil jika wajah hilang dari frame."""
        self._eye_buf.clear()
        self._mouth_buf.clear()

    @property
    def perclos_history(self) -> list:
        """Untuk plot grafik PERCLOS real-time di Streamlit."""
        n = len(self._eye_buf)
        history = []
        buf = list(self._eye_buf)
        for i in range(1, n + 1):
            history.append(sum(buf[:i]) / i)
        return history

    @property
    def threshold(self) -> float:
        return self.perclos_th
