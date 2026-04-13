# config.py
# Semua konfigurasi sistem di satu tempat.
# Jika ada yang perlu diubah, cukup ubah di file ini saja.

import os

# ── PATH MODEL ────────────────────────────────────────────────
# Taruh file .keras di folder models/
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATHS = {
    "InceptionV3" : os.path.join(MODELS_DIR, "InceptionV3_after_finetune.keras"),
    "MobileNetV2" : os.path.join(MODELS_DIR, "MobileNetV2_after_finetune.keras"),
    "ResNet50V2"  : os.path.join(MODELS_DIR, "ResNet50V2_after_finetune.keras"),
}

# ── KELAS ─────────────────────────────────────────────────────
CLASS_NAMES   = ["open_eye", "closed_eye", "yawn"]
NUM_CLASSES   = len(CLASS_NAMES)

# ── PREPROCESSING ─────────────────────────────────────────────
IMG_SIZE      = (96, 96)          # ukuran input ke semua model
CLAHE_CLIP    = 2.0               # clipLimit CLAHE
CLAHE_GRID    = (4, 4)            # tileGridSize CLAHE

# ── MEDIAPIPE LANDMARK INDICES ────────────────────────────────
# Indeks landmark FaceMesh 468 titik untuk setiap ROI
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX     = [61, 185, 40, 39, 37, 267, 269, 270, 409, 291]

ROI_PADDING_EYE   = 0.25
ROI_PADDING_MOUTH = 0.35

# ── PERCLOS ───────────────────────────────────────────────────
# Referensi: NHTSA PERCLOS standard
# Kantuk = mata tertutup >= 70% dalam window 30 frame terakhir
PERCLOS_WINDOW    = 30            # jumlah frame window
PERCLOS_THRESHOLD = 0.70          # 70% → kantuk
YAWN_THRESHOLD    = 2             # jumlah menguap dalam window → kantuk
CONFIDENCE_MIN    = 0.55          # prediksi di bawah ini diabaikan

# ── ALARM ─────────────────────────────────────────────────────
ALARM_COOLDOWN    = 4.0           # detik jeda antar alarm
ALARM_FREQ_START  = 440           # Hz — frekuensi awal chirp
ALARM_FREQ_END    = 880           # Hz — frekuensi akhir chirp
ALARM_DURATION    = 1.2           # detik
ALARM_SAMPLE_RATE = 44100

# ── MEDIAPIPE ─────────────────────────────────────────────────
FACE_DETECTION_CONF  = 0.5
FACE_TRACKING_CONF   = 0.5

# ── DISPLAY ───────────────────────────────────────────────────
STATUS_COLORS = {
    "normal"  : (34, 197, 94),    # hijau — RGB untuk Streamlit
    "drowsy"  : (239, 68, 68),    # merah
    "warning" : (251, 146, 60),   # oranye
}
