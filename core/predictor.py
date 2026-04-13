# core/predictor.py
# Tanggung jawab: load 3 model CNN dan menggabungkan
# prediksinya dengan weighted voting berdasarkan val_accuracy.

import os
import numpy as np
import tensorflow as tf
from typing import Optional
from config import MODEL_PATHS, CLASS_NAMES, NUM_CLASSES, CONFIDENCE_MIN

import os
import gdown

import requests

def download_file(file_id, output_path):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

def download_models():
    os.makedirs("models", exist_ok=True)

    models = {
        "InceptionV3_after_finetune.keras": "1a3nMOyBPc1YlUmhfvcOcVEYnqMYiQYws",
        "MobileNetV2_after_finetune.keras": "1qRdcQwOBx9Vk3kYTDysQU9-0bNJ0SIHt",
        "ResNet50V2_after_finetune.keras": "1xwOmXfvDz44fEeo6h1xX9FfBNQh4o_IL",
    }

    for filename, file_id in models.items():
        path = os.path.join("models", filename)

        if not os.path.exists(path):
            print(f"Downloading {filename}...")
            download_file(file_id, path)

            if not os.path.exists(path):
                raise RuntimeError(f"Gagal download {filename}")
class DrowsinessPredictor:
    """
    Mengelola 3 model CNN dan melakukan weighted voting.

    Weighted voting:
    - Bobot setiap model = val_accuracy-nya (normalized)
    - Model yang lebih akurat di validasi mendapat suara lebih besar
    - Ini lebih baik dari majority vote karena memperhitungkan
      kepercayaan relatif antar model
    """

    def __init__(self, val_accuracies: Optional[dict] = None):
        """
        val_accuracies: dict {model_name: accuracy_float_0_to_100}
        Jika None, semua model mendapat bobot sama (1/3 masing-masing).
        """
        self.models   = {}
        self.weights  = {}
        download_models()
        self._load_models(val_accuracies)

    def _load_models(self, val_accuracies: Optional[dict]):
        """Load semua model .keras dari folder models/."""
        print("Memuat model CNN...")
        loaded_accs = {}

        for name, path in MODEL_PATHS.items():
            if not os.path.exists(path):
                print(f"  [SKIP] {name}: file tidak ditemukan di {path}")
                continue
            try:
                self.models[name] = tf.keras.models.load_model(path)
                acc = (val_accuracies or {}).get(name, 100.0)
                loaded_accs[name] = acc
                print(f"  [OK]   {name} (val_acc={acc:.1f}%)")
            except Exception as e:
                print(f"  [ERROR] {name}: {e}")

        if not self.models:
            raise RuntimeError(
                "Tidak ada model yang berhasil dimuat. "
                "Pastikan file .keras ada di folder models/."
            )

        # Hitung bobot ternormalisasi
        total = sum(loaded_accs.values())
        self.weights = {
            name: acc / total
            for name, acc in loaded_accs.items()
        }
        print(f"\nBobot voting:")
        for name, w in self.weights.items():
            print(f"  {name}: {w:.3f}")

    def predict(self, roi_input: np.ndarray) -> dict:
        """
        Prediksi satu ROI dengan semua model, gabungkan dengan
        weighted voting.

        Args:
            roi_input: array shape (1, 96, 96, 3), sudah dinormalisasi

        Returns dict:
            class_name   : str — hasil prediksi akhir
            confidence   : float — kepercayaan prediksi [0,1]
            probabilities: list — probabilitas per kelas setelah voting
            per_model    : dict — prediksi dan confidence tiap model
            is_reliable  : bool — False jika confidence < CONFIDENCE_MIN
        """
        combined   = np.zeros(NUM_CLASSES)
        per_model  = {}

        for name, model in self.models.items():
            prob = model.predict(roi_input, verbose=0)[0]
            w    = self.weights.get(name, 1.0 / len(self.models))
            combined += w * prob

            pred_idx = np.argmax(prob)
            per_model[name] = {
                "class"     : CLASS_NAMES[pred_idx],
                "confidence": float(prob[pred_idx]),
                "probs"     : prob.tolist(),
            }

        pred_idx   = int(np.argmax(combined))
        confidence = float(combined[pred_idx])

        return {
            "class_name"   : CLASS_NAMES[pred_idx],
            "confidence"   : confidence,
            "probabilities": combined.tolist(),
            "per_model"    : per_model,
            "is_reliable"  : confidence >= CONFIDENCE_MIN,
        }

    @property
    def model_names(self) -> list:
        return list(self.models.keys())

    @property
    def n_models(self) -> int:
        return len(self.models)
