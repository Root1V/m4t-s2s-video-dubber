"""Extrae la pista de audio de un video y la guarda como WAV.

Uso:
  python tools/extraer_audio.py                              # INPUT_DIR/video.mp4 → audio_original.wav
  python tools/extraer_audio.py /ruta/video.mp4              # video específico
  python tools/extraer_audio.py video.mp4 -o salida.wav      # nombre de salida
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torchaudio

from m4t_dubber import config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrae audio de un video a WAV.")
    parser.add_argument("video", nargs="?", help="Ruta al video (default: INPUT_DIR/video.mp4)")
    parser.add_argument("-o", "--output", default="audio_original.wav", help="Archivo WAV de salida")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    video_path = Path(args.video) if args.video else config.INPUT_DIR / "video.mp4"

    if not video_path.exists():
        print(f"❌ Error: No se encontró '{video_path}'")
        sys.exit(1)

    out_path = Path(args.output)
    print(f"📦 Extrayendo audio de: {video_path}")

    try:
        audio, orig_freq = torchaudio.load(str(video_path))
        print(f"   ✓ Audio cargado — Shape: {audio.shape}, Frecuencia: {orig_freq}Hz")
        print(f"   ✓ Rango: [{audio.min():.4f}, {audio.max():.4f}]")

        torchaudio.save(str(out_path), audio, orig_freq)
        print(f"\n✅ Audio guardado: '{out_path}'")
        print(f"   Duración: {audio.shape[1] / orig_freq:.2f}s")

    except Exception as e:
        print(f"❌ Error al extraer audio: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
