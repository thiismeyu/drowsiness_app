# Deteksi Kantuk Pengendara
Komparasi InceptionV3 · MobileNetV2 · ResNet50V2 dengan MediaPipe + PERCLOS

## Struktur project

```
drowsiness_app/
├── models/                        ← taruh file .keras di sini
│   ├── inceptionv3_final.keras
│   ├── mobilenet_final.keras
│   └── resnet_final.keras
├── core/
│   ├── detector.py                ← MediaPipe FaceMesh + crop ROI
│   ├── predictor.py               ← load model + weighted voting
│   └── perclos.py                 ← logika PERCLOS + keputusan kantuk
├── alarm/
│   └── alarm.py                   ← generate alarm (tanpa file eksternal)
├── .streamlit/
│   └── config.toml                ← tema Streamlit
├── config.py                      ← semua konfigurasi di satu tempat
├── app.py                         ← entry point Streamlit
└── requirements.txt
```

## Setup lokal (VS Code)

```bash
# 1. Buat virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 2. Install dependensi
pip install -r requirements.txt

# 3. Taruh file model
# Salin file .keras dari Google Drive ke folder models/
# Pastikan nama file sesuai dengan MODEL_PATHS di config.py

# 4. Jalankan
streamlit run app.py
```

Buka browser di http://localhost:8501

## Konfigurasi val accuracy model

Buka sidebar di aplikasi dan isi val accuracy dari hasil
evaluasi notebook Colab Anda. Ini dipakai untuk menentukan
bobot voting setiap model — model yang lebih akurat
mendapat suara lebih besar.

Atau edit langsung di `app.py` bagian `init_session`:
```python
"val_accuracies": {
    "InceptionV3" : 95.0,   # ganti dengan nilai dari Colab
    "MobileNetV2" : 93.0,
    "ResNet50V2"  : 94.0,
},
```

## Cara kerja sistem

```
Kamera
  ↓
MediaPipe FaceMesh (468 landmark)
  ↓
Crop ROI: mata kiri · mata kanan · mulut
  ↓
CLAHE preprocessing (sama persis dengan training)
  ↓
3 CNN prediksi paralel
  ↓
Weighted voting (bobot = val accuracy)
  ↓
PERCLOS counter (window 30 frame)
  ↓
Keputusan kantuk + alarm
```

## Mengubah threshold

Di sidebar aplikasi, Anda bisa ubah:
- **Threshold PERCLOS**: persentase frame mata tertutup (default 70%)
- **Threshold menguap**: jumlah frame menguap dalam window (default 2)

Atau ubah default di `config.py`:
```python
PERCLOS_THRESHOLD = 0.70
YAWN_THRESHOLD    = 2
```
