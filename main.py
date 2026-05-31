"""CLI entrypoint para el dubbing pipeline.

Uso:
  python main.py                  # traduce todos los videos en INPUT_DIR
  python main.py tutorial.mp4     # traduce solo ese video
  python main.py /ruta/video.mkv  # ruta absoluta

Variables de entorno (o archivo .env):
  M4T_INPUT_DIR      Carpeta de videos de entrada
  M4T_OUTPUT_DIR     Carpeta de resultados
  M4T_PROCESSED_DIR  Carpeta para videos ya procesados
  M4T_SPEAKER_ID     ID de voz del vocoder (0-199; default 4)
  M4T_TGT_LANG       Idioma destino (default: spa)
"""

import argparse
import sys

from m4t_dubber.pipeline import DubbingPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="M4T Video Dubber — traduce videos de inglés a español.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "video",
        nargs="?",
        metavar="VIDEO",
        help="Video a procesar (opcional; si se omite se procesan todos en INPUT_DIR)",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    pipeline = DubbingPipeline()
    pipeline.run(args.video)
    return 0


if __name__ == "__main__":
    sys.exit(main())
