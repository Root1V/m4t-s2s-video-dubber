"""Configuración centralizada — sobreescribible con variables de entorno.

Copia .env.example como .env y ajusta las rutas antes de ejecutar.
"""

import os
from pathlib import Path

# ── Rutas ─────────────────────────────────────────────────────────
# La raíz del proyecto es el directorio padre del repo (video_translate/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_DIR     = Path(os.getenv("M4T_INPUT_DIR",     str(_PROJECT_ROOT / "videos")))
OUTPUT_DIR    = Path(os.getenv("M4T_OUTPUT_DIR",    str(_PROJECT_ROOT / "resultados")))
PROCESSED_DIR = Path(os.getenv("M4T_PROCESSED_DIR", str(_PROJECT_ROOT / "procesados")))

# ── Modelo ────────────────────────────────────────────────────────
MODEL_ID   = os.getenv("M4T_MODEL_ID",  "facebook/seamless-m4t-v2-large")
SRC_LANG   = os.getenv("M4T_SRC_LANG", "eng")  # idioma fuente (SeamlessM4T auto-detecta si no se especifica)
TGT_LANG   = os.getenv("M4T_TGT_LANG", "spa")
SPEAKER_ID = int(os.getenv("M4T_SPEAKER_ID", "4"))  # 0-199; 4 = voz masculina en español

# ── Parámetros de generación ──────────────────────────────────────
NO_REPEAT_NGRAM_SIZE = int(os.getenv("M4T_NO_REPEAT_NGRAM_SIZE", "5"))
REPETITION_PENALTY   = float(os.getenv("M4T_REPETITION_PENALTY", "1.3"))
NUM_BEAMS            = int(os.getenv("M4T_NUM_BEAMS", "2"))

# ── Límites de procesamiento (MPS Apple Silicon) ──────────────────
# MPS tiene un límite INT_MAX; >15s a 16kHz puede desbordarlo
MAX_CHUNK_S = int(os.getenv("M4T_MAX_CHUNK_S", "15"))
MIN_CHUNK_S = float(os.getenv("M4T_MIN_CHUNK_S", "1"))

# ── VAD (detección de actividad de voz) ──────────────────────────
VAD_FRAME_MS       = int(os.getenv("M4T_VAD_FRAME_MS",       "20"))
VAD_MIN_SILENCE_MS = int(os.getenv("M4T_VAD_MIN_SILENCE_MS", "400"))
VAD_PERCENTILE     = int(os.getenv("M4T_VAD_PERCENTILE",     "20"))
VAD_MULTIPLIER     = int(os.getenv("M4T_VAD_MULTIPLIER",     "3"))

# ── Constantes ────────────────────────────────────────────────────
SAMPLE_RATE       = 16_000  # SeamlessM4T requiere 16 kHz
VIDEO_EXTENSIONS  = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v")
