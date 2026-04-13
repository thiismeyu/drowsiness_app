# core/detector.py
# Tanggung jawab: deteksi wajah dengan MediaPipe FaceMesh
# dan crop ROI (mata kiri, mata kanan, mulut) dari setiap frame.

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple
from config import (
    LEFT_EYE_IDX, RIGHT_EYE_IDX, MOUTH_IDX,
    ROI_PADDING_EYE, ROI_PADDING_MOUTH,
    IMG_SIZE, CLAHE_CLIP, CLAHE_GRID,
    FACE_DETECTION_CONF, FACE_TRACKING_CONF,
)


class FaceROIDetector:
    """
    Mendeteksi wajah dan mengekstrak ROI mata + mulut
    menggunakan MediaPipe FaceMesh.

    Mengapa MediaPipe bukan Haar Cascade atau MTCNN:
    - 468 landmark presisi, stabil di berbagai pose kepala
    - Berjalan real-time (30+ FPS) tanpa GPU
    - Gratis, tidak perlu training tambahan
    """

    def __init__(self):
        self._mp_face = mp.solutions.face_mesh
        self._face_mesh = self._mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=FACE_DETECTION_CONF,
            min_tracking_confidence=FACE_TRACKING_CONF,
        )

    def detect(self, frame_bgr: np.ndarray) -> dict:
        """
        Proses satu frame BGR.

        Returns dict:
            face_detected : bool
            rois          : dict dengan key "left_eye", "right_eye", "mouth"
                            masing-masing berisi (roi_array, bbox_tuple) atau None
            landmarks     : list landmark raw (untuk debugging)
            face_bbox     : (x1,y1,x2,y2) bounding box wajah keseluruhan
        """
        result = {
            "face_detected": False,
            "rois"         : {"left_eye": None, "right_eye": None, "mouth": None},
            "landmarks"    : None,
            "face_bbox"    : None,
        }

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_result = self._face_mesh.process(rgb)

        if not mp_result.multi_face_landmarks:
            return result

        result["face_detected"] = True
        lm = mp_result.multi_face_landmarks[0].landmark
        result["landmarks"] = lm

        # Bounding box wajah keseluruhan
        h, w = frame_bgr.shape[:2]
        all_x = [p.x * w for p in lm]
        all_y = [p.y * h for p in lm]
        result["face_bbox"] = (
            int(min(all_x)), int(min(all_y)),
            int(max(all_x)), int(max(all_y))
        )

        # Crop ROI masing-masing region
        result["rois"]["left_eye"]  = self._crop_roi(
            frame_bgr, lm, LEFT_EYE_IDX,  ROI_PADDING_EYE)
        result["rois"]["right_eye"] = self._crop_roi(
            frame_bgr, lm, RIGHT_EYE_IDX, ROI_PADDING_EYE)
        result["rois"]["mouth"]     = self._crop_roi(
            frame_bgr, lm, MOUTH_IDX,     ROI_PADDING_MOUTH)

        return result

    def _crop_roi(
        self,
        frame   : np.ndarray,
        lm      : list,
        indices : list,
        padding : float,
    ) -> Optional[Tuple[np.ndarray, tuple]]:
        """
        Crop area dari frame berdasarkan indeks landmark.
        Tambahkan padding proporsional agar ROI tidak terlalu ketat.
        Return (roi_bgr, (x1, y1, x2, y2)) atau None jika gagal.
        """
        h, w = frame.shape[:2]
        xs = [lm[i].x * w for i in indices]
        ys = [lm[i].y * h for i in indices]

        rng_x = max(xs) - min(xs)
        rng_y = max(ys) - min(ys)

        x1 = max(0, int(min(xs) - padding * rng_x))
        x2 = min(w, int(max(xs) + padding * rng_x))
        y1 = max(0, int(min(ys) - padding * rng_y))
        y2 = min(h, int(max(ys) + padding * rng_y))

        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        return roi, (x1, y1, x2, y2)

    def close(self):
        self._face_mesh.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def preprocess_roi(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocessing ROI sebelum masuk model CNN:
    1. CLAHE — perbaiki kontras, terutama untuk kabin gelap/terang
    2. Resize ke IMG_SIZE
    3. Normalize ke [0, 1]
    4. Tambah batch dimension

    WAJIB sama persis dengan preprocessing saat training di Colab.
    """
    # CLAHE di ruang LAB (hanya channel L = brightness)
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    lab = cv2.merge([clahe.apply(l), a, b])
    roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Resize + normalize
    roi = cv2.resize(roi, IMG_SIZE).astype(np.float32) / 255.0
    return np.expand_dims(roi, axis=0)   # shape: (1, 96, 96, 3)
