"""Prueba diferentes speaker_id del vocoder para encontrar la voz deseada.

Genera un WAV por cada ID con los primeros N segundos del video.
Escucha los archivos y apunta el ID que suene mejor.
Luego configura M4T_SPEAKER_ID=<id> en tu .env (o en config.py).

Uso:
  python tools/probar_voces.py                        # usa INPUT_DIR/video.mp4
  python tools/probar_voces.py /ruta/al/video.mp4     # video específico
  python tools/probar_voces.py video.mp4 --ids 0-20   # rango de IDs
  python tools/probar_voces.py video.mp4 --segundos 5 # fragmento de 5s
"""

import argparse
import sys
from pathlib import Path

# Permite importar m4t_dubber aunque se ejecute desde tools/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToSpeech

from m4t_dubber import config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prueba speaker IDs del vocoder.")
    parser.add_argument("video", nargs="?", help="Ruta al video (default: INPUT_DIR/video.mp4)")
    parser.add_argument(
        "--ids", default="0-9",
        help="Rango de IDs a probar, ej: '0-9' o '0-19' (default: 0-9)",
    )
    parser.add_argument("--segundos", type=int, default=10, help="Segundos de audio a usar")
    return parser.parse_args()


def _parse_range(s: str) -> list[int]:
    lo, hi = s.split("-")
    return list(range(int(lo), int(hi) + 1))

def main() -> None:
    args = _parse_args()
    ids_a_probar = _parse_range(args.ids)
    segundos     = args.segundos

    video_path = (
        Path(args.video)
        if args.video
        else config.INPUT_DIR / "video.mp4"
    )
    if not video_path.exists():
        print(f"❌ No se encontró '{video_path}'.")
        sys.exit(1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🍏 Procesando en: {device}")

    print(f"🤖 Cargando {config.MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(config.MODEL_ID)
    model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(config.MODEL_ID).to(device)

    print(f"📦 Cargando audio de: {video_path}")
    audio, orig_freq = torchaudio.load(str(video_path))
    if orig_freq != config.SAMPLE_RATE:
        audio = torchaudio.transforms.Resample(orig_freq, config.SAMPLE_RATE)(audio)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    muestra = audio[:, : config.SAMPLE_RATE * segundos]

    audio_inputs = processor(
        audio=muestra.squeeze(0).numpy(),
        sampling_rate=config.SAMPLE_RATE,
        return_tensors="pt",
    ).to(device)

    out_dir = Path(".")  # guarda en el directorio actual
    print(f"\n🎙️  Generando {len(ids_a_probar)} voces ({segundos}s cada una) → {out_dir.resolve()}\n")

    for sid in ids_a_probar:
        with torch.no_grad():
            output = model.generate(
                **audio_inputs,
                tgt_lang=config.TGT_LANG,
                speaker_id=sid,
                no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
                repetition_penalty=config.REPETITION_PENALTY,
            )
        audio_out = (output[0] if isinstance(output, tuple) else output).cpu()
        if audio_out.dim() == 1:
            audio_out = audio_out.unsqueeze(0)

        nombre = out_dir / f"voz_id_{sid:03d}.wav"
        torchaudio.save(str(nombre), audio_out, config.SAMPLE_RATE)
        print(f"   ✓ speaker_id={sid:3d}  →  {nombre}")

        if device.type == "mps":
            torch.mps.empty_cache()

    print(f"\n✅ Listo. Escucha los archivos 'voz_id_*.wav' y apunta el ID masculino.")
    print("   Luego en traducir.py cambia:")
    print('       SPEAKER_GENDER = "male"')
    print("   y agrega una línea debajo de esa asignación:")
    print('       # speaker_id = <el número que encontraste>')
    print("   y en la sección de selección de speaker, ajusta el speaker_id al valor correcto.")

if __name__ == "__main__":
    main()
