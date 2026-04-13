import os
import gdown
import numpy as np
import tensorflow as tf
from typing import Optional
from config import MODEL_PATHS, CLASS_NAMES, NUM_CLASSES, CONFIDENCE_MIN


def download_models():
    import os
    import requests

    os.makedirs("models", exist_ok=True)

    models = {
        "InceptionV3_after_finetune.h5": "https://huggingface.co/ayuuuuuuu/drowsiness-model/resolve/main/InceptionV3_after_finetune.h5",
        "MobileNetV2_after_finetune.h5": "https://huggingface.co/ayuuuuuuu/drowsiness-model/resolve/main/MobileNetV2_after_finetune.h5",
        "ResNet50V2_after_finetune.h5": "https://huggingface.co/ayuuuuuuu/drowsiness-model/resolve/main/ResNet50V2_after_finetune.h5",
    }

    for filename, url in models.items():
        path = os.path.join("models", filename)

        if not os.path.exists(path):
            print(f"Downloading {filename}...")

            try:
                r = requests.get(url, stream=True)
                if r.status_code != 200:
                    print(f"Gagal download {filename}, status:", r.status_code)
                    continue

                with open(path, "wb") as f:
                    for chunk in r.iter_content(1024):
                        if chunk:
                            f.write(chunk)

                print(f"{filename} selesai ✔ size:", os.path.getsize(path))

            except Exception as e:
                print(f"Gagal download {filename}: {e}")

    print("ISI MODELS:", os.listdir("models"))


class DrowsinessPredictor:

    def __init__(self, val_accuracies: Optional[dict] = None):
        self.models = {}
        self.weights = {}

        download_models()
        self._load_models(val_accuracies)

    def _load_models(self, val_accuracies):
        print("CEK FILE:", path, os.path.exists(path))
        print("ISI MODELS:", os.listdir("models"))

        print("Memuat model CNN...")
        loaded_accs = {}
        

        for name, path in MODEL_PATHS.items():

            if not os.path.exists(path):
                print(f"[SKIP] {name} tidak ditemukan")
                continue

            try:
                model = tf.keras.models.load_model(
                    path,
                    compile=False,
                    safe_mode=False
                )

                self.models[name] = model

                acc = (val_accuracies or {}).get(name, 100.0)
                loaded_accs[name] = acc

                print(f"[OK] {name} loaded")

            except Exception as e:
                print(f"[ERROR] {name}: {e}")

        if not self.models:
            raise RuntimeError("Tidak ada model berhasil dimuat")

        total = sum(loaded_accs.values())
        self.weights = {
            name: acc / total for name, acc in loaded_accs.items()
        }

    def predict(self, roi_input):

        combined = np.zeros(NUM_CLASSES)
        per_model = {}

        for name, model in self.models.items():

            prob = model.predict(roi_input, verbose=0)[0]
            w = self.weights.get(name, 1 / len(self.models))

            combined += w * prob

            pred_idx = np.argmax(prob)

            per_model[name] = {
                "class": CLASS_NAMES[pred_idx],
                "confidence": float(prob[pred_idx])
            }

        pred_idx = np.argmax(combined)

        return {
            "class_name": CLASS_NAMES[pred_idx],
            "confidence": float(combined[pred_idx]),
            "per_model": per_model,
            "is_reliable": combined[pred_idx] >= CONFIDENCE_MIN
        }