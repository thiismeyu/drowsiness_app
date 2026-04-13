# alarm/alarm.py
# Generate dan putar alarm audio tanpa file eksternal.
# Menggunakan scipy untuk membuat gelombang chirp (sweep frekuensi),
# lalu base64-encode untuk dikirim ke browser via Streamlit.

import time
import base64
import io
import numpy as np
from scipy.io.wavfile import write as wav_write
from scipy.signal import chirp
from config import (
    ALARM_FREQ_START, ALARM_FREQ_END,
    ALARM_DURATION, ALARM_SAMPLE_RATE,
    ALARM_COOLDOWN,
)


class AlarmManager:
    """
    Mengelola pembuatan dan pemutaran alarm audio.

    Alarm dibuat secara programatik (chirp 440→880 Hz) sehingga
    tidak memerlukan file .mp3 atau .wav eksternal.
    Di Streamlit, audio dikirim sebagai base64 HTML audio tag.
    """

    def __init__(self):
        self._last_alarm_time = 0.0
        self._wav_bytes       = self._generate_wav()
        self._wav_b64         = base64.b64encode(
            self._wav_bytes).decode("utf-8")

    def _generate_wav(self) -> bytes:
        """
        Buat gelombang chirp (sweep frekuensi) sebagai WAV bytes.
        Chirp terdengar lebih mencolok dari beep biasa —
        variasi frekuensi membuatnya lebih sulit diabaikan.
        """
        t = np.linspace(0, ALARM_DURATION,
                        int(ALARM_SAMPLE_RATE * ALARM_DURATION))
        wave = chirp(t,
                     f0=ALARM_FREQ_START,
                     f1=ALARM_FREQ_END,
                     t1=ALARM_DURATION,
                     method="linear")

        # Fade in/out 50ms agar tidak ada klik di awal/akhir
        fade_samples = int(ALARM_SAMPLE_RATE * 0.05)
        fade_in  = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        wave[:fade_samples]  *= fade_in
        wave[-fade_samples:] *= fade_out

        wave_int16 = (wave * 32767 * 0.85).astype(np.int16)

        buf = io.BytesIO()
        wav_write(buf, ALARM_SAMPLE_RATE, wave_int16)
        return buf.getvalue()

    def should_alarm(self) -> bool:
        """Cek apakah sudah melewati cooldown."""
        return (time.time() - self._last_alarm_time) >= ALARM_COOLDOWN

    def get_html_audio_tag(self) -> str:
        """
        Kembalikan HTML <audio> tag yang langsung autoplay.
        Dipakai oleh Streamlit lewat st.markdown(..., unsafe_allow_html=True).
        """
        self._last_alarm_time = time.time()
        return f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{self._wav_b64}"
                    type="audio/wav">
        </audio>
        """

    def trigger(self) -> str:
        """
        Trigger alarm jika cooldown sudah lewat.
        Return HTML audio tag jika alarm berbunyi, string kosong jika tidak.
        """
        if self.should_alarm():
            return self.get_html_audio_tag()
        return ""

    @property
    def cooldown_remaining(self) -> float:
        """Sisa waktu cooldown dalam detik."""
        elapsed = time.time() - self._last_alarm_time
        return max(0.0, ALARM_COOLDOWN - elapsed)
