"""CLI entrypoint para el dubbing pipeline.

Uso:
  python main.py                               # traduce todos los videos en INPUT_DIR
  python main.py tutorial.mp4                  # traduce solo ese video
  python main.py tutorial.mp4 --tgt-lang fra    # traducir al francés
  python main.py tutorial.mp4 --src-lang spa --tgt-lang eng  # español → inglés
  python main.py tutorial.mp4 --srt             # generar subtítulos .srt además del video
  python main.py tutorial.mp4 --stem            # separar voz de fondo antes de traducir
  python main.py tutorial.mp4 --stem --srt      # separar voz + generar subtítulos
  python main.py tutorial.mp4 --clone-voice     # clonar voz del hablante original con F5-TTS
  python main.py tutorial.mp4 --clone-voice --stem  # clonar voz + preservar música de fondo

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
    parser.add_argument(
        "--srt",
        action="store_true",
        help="Generar archivo de subtítulos .srt junto al video traducido",
    )
    parser.add_argument(
        "--stem",
        action="store_true",
        help=(
            "Separar voz de fondo con Demucs (htdemucs) antes de traducir. "
            "Preserva música y efectos de audio intactos. Requiere ~80 MB extra."
        ),
    )
    parser.add_argument(
        "--clone-voice",
        action="store_true",
        dest="clone_voice",
        help=(
            "Clonar la voz del hablante original con F5-TTS (zero-shot). "
            "La síntesis usa los primeros 15 s del audio como referencia. "
            "Requiere ~1.5 GB de descarga en primer uso. "
            "Combinar con --stem para preservar también la música de fondo."
        ),
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    pipeline = DubbingPipeline()
    pipeline.run(
        args.video,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        generate_srt=args.srt,
        use_stem=args.stem,
        clone_voice=args.clone_voice,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
