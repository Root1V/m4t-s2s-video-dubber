"""CLI entrypoint para el dubbing pipeline.

Uso:
  python main.py                               # traduce todos los videos en INPUT_DIR
  python main.py tutorial.mp4                  # traduce solo ese video
  python main.py tutorial.mp4 --tgt-lang fra   # traducir al francés
  python main.py tutorial.mp4 --src-lang spa --tgt-lang eng  # español → inglés

Variables de entorno (o archivo .env):
  M4T_INPUT_DIR      Carpeta de videos de entrada
  M4T_OUTPUT_DIR     Carpeta de resultados
  M4T_PROCESSED_DIR  Carpeta para videos ya procesados
  M4T_SPEAKER_ID     ID de voz del vocoder (0-199; default 4)
  M4T_SRC_LANG       Idioma fuente (default: eng)
  M4T_TGT_LANG       Idioma destino (default: spa)

Idiomas soportados (ejemplos): eng, spa, fra, deu, por, ita, jpn, cmn, ara, rus
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
    parser.add_argument(
        "--src-lang",
        default=None,
        metavar="LANG",
        help="Idioma fuente (ej: eng, spa, fra). Default: valor de M4T_SRC_LANG o 'eng'",
    )
    parser.add_argument(
        "--tgt-lang",
        default=None,
        metavar="LANG",
        help="Idioma destino (ej: spa, fra, por, deu). Default: valor de M4T_TGT_LANG o 'spa'",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    pipeline = DubbingPipeline()
    pipeline.run(args.video, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
    return 0


if __name__ == "__main__":
    sys.exit(main())
