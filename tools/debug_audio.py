"""Diagnóstico del pipeline de traducción — procesa 10s de un video y valida la salida.

Uso:
  python tools/debug_audio.py                      # usa INPUT_DIR/video.mp4
  python tools/debug_audio.py /ruta/al/video.mp4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import SeamlessM4Tv2ForSpeechToSpeech, AutoProcessor
import torchaudio
import numpy as np

from m4t_dubber import config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnóstico del pipeline M4T.")
    parser.add_argument("video", nargs="?", help="Ruta al video (default: INPUT_DIR/video.mp4)")
    parser.add_argument("-o", "--output", default="audio_prueba_traducido.wav")
    return parser.parse_args()


# --- CONFIGURACIÓN DE ARCHIVOS ---
ARCHIVO_ENTRADA = "video.mp4"
# ---------------------------------

def debug_audio_translation(video_path: Path, output_path: Path) -> None:
    if not video_path.exists():
        print(f"❌ Error: No se encontró '{video_path}'")
        sys.exit(1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🍏 Device: {device}\n")

    # Cargar modelo
    print("📥 Cargando modelo...")
    processor = AutoProcessor.from_pretrained(config.MODEL_ID)
    model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(config.MODEL_ID).to(device)

    # Cargar audio original
    print("📦 Cargando audio del video...")
    audio, orig_freq = torchaudio.load(str(video_path))
    print(f"   ✓ Audio original - Shape: {audio.shape}, Frecuencia: {orig_freq}Hz")
    print(f"   ✓ Rango de valores: [{audio.min():.4f}, {audio.max():.4f}]")
    print(f"   ✓ Dtype: {audio.dtype}")

    # Remuestrear
    if orig_freq != config.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq, config.SAMPLE_RATE)
        audio = resampler(audio)
        print(f"   ✓ Audio remuestreado a {config.SAMPLE_RATE}Hz - Shape: {audio.shape}")

    # Convertir a mono
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
        print(f"   ✓ Audio convertido a mono - Shape: {audio.shape}")

    # Procesar un pequeño fragmento de PRUEBA (10 segundos)
    print("\n🧪 PRUEBA: Procesando 10 segundos de audio...")
    fragmento_prueba = audio[:, : config.SAMPLE_RATE * 10]
    print(f"   ✓ Fragmento de prueba - Shape: {fragmento_prueba.shape}")
    print(f"   ✓ Rango: [{fragmento_prueba.min():.4f}, {fragmento_prueba.max():.4f}]")

    # Preparar entrada
    print("\n🔧 Preparando entrada del procesador...")
    audio_inputs = processor(
        audio=fragmento_prueba.squeeze(0).numpy(),
        sampling_rate=config.SAMPLE_RATE,
        return_tensors="pt"
    )
    print(f"   ✓ Claves en audio_inputs: {audio_inputs.keys()}")
    for key in audio_inputs:
        if hasattr(audio_inputs[key], 'shape'):
            print(f"     - {key}: shape={audio_inputs[key].shape}, dtype={audio_inputs[key].dtype}")

    # Mover a device
    audio_inputs = audio_inputs.to(device)

    # Generar traducción con los mismos parámetros que traducir.py
    print("\n🗣️ Generando traducción al español...")
    with torch.no_grad():
        output = model.generate(
            **audio_inputs,
            tgt_lang=config.TGT_LANG,
            no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
            repetition_penalty=config.REPETITION_PENALTY,
            num_beams=config.NUM_BEAMS,
        )

    # Analizar salida
    print("\n📊 ANÁLISIS DE SALIDA:")
    print(f"   ✓ Tipo de output: {type(output)}")
    
    if isinstance(output, tuple):
        print(f"   ✓ Es una tupla con {len(output)} elementos")
        for idx, item in enumerate(output):
            if isinstance(item, torch.Tensor):
                print(f"     [{idx}] Tensor - Shape: {item.shape}, Dtype: {item.dtype}, "
                      f"Rango: [{item.min():.6f}, {item.max():.6f}]")
            else:
                print(f"     [{idx}] {type(item)}: {item}")
        
        # Extraer tensor de audio
        audio_tensor = output[0]
    else:
        print(f"   ✓ Tensor directo - Shape: {output.shape}, Dtype: {output.dtype}")
        print(f"     Rango: [{output.min():.6f}, {output.max():.6f}]")
        audio_tensor = output

    print(f"\n✅ Audio de salida final:")
    print(f"   - Shape: {audio_tensor.shape}")
    print(f"   - Dtype: {audio_tensor.dtype}")
    print(f"   - Rango: [{audio_tensor.min():.6f}, {audio_tensor.max():.6f}]")
    print(f"   - Media: {audio_tensor.mean():.6f}")
    print(f"   - Desv. Est.: {audio_tensor.std():.6f}")

    # Guardar audio de prueba
    print(f"\n💾 Guardando audio de prueba...")
    audio_tensor_cpu = audio_tensor.cpu()
    
    # Normalizar si es necesario
    max_val = audio_tensor_cpu.abs().max()
    if max_val > 0:
        audio_tensor_cpu = audio_tensor_cpu / max_val
        print(f"   ✓ Audio normalizado (máximo valor era {max_val:.6f})")

    # Asegurar que sea (1, samples)
    if len(audio_tensor_cpu.shape) == 1:
        audio_tensor_cpu = audio_tensor_cpu.unsqueeze(0)
        print(f"   ✓ Audio reshape a: {audio_tensor_cpu.shape}")

    torchaudio.save(str(output_path), audio_tensor_cpu, config.SAMPLE_RATE)
    print(f"   ✓ Guardado: {output_path}")

    print("\n✅ Debug completado. Escucha el archivo de salida para validar la calidad.")


if __name__ == "__main__":
    _args = _parse_args()
    _video = Path(_args.video) if _args.video else config.INPUT_DIR / "video.mp4"
    debug_audio_translation(_video, Path(_args.output))

if __name__ == "__main__":
    debug_audio_translation()
