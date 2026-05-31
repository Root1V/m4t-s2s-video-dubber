import torch
from transformers import SeamlessM4Tv2ForSpeechToSpeech, AutoProcessor
import torchaudio
import os
import numpy as np

# --- CONFIGURACIÓN DE ARCHIVOS ---
ARCHIVO_ENTRADA = "video.mp4"
# ---------------------------------

def debug_audio_translation():
    if not os.path.exists(ARCHIVO_ENTRADA):
        print(f"❌ Error: No se encontró '{ARCHIVO_ENTRADA}'")
        return

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🍏 Device: {device}\n")

    # Cargar modelo
    print("📥 Cargando modelo...")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained("facebook/seamless-m4t-v2-large").to(device)

    # Cargar audio original
    print("📦 Cargando audio del video...")
    audio, orig_freq = torchaudio.load(ARCHIVO_ENTRADA)
    print(f"   ✓ Audio original - Shape: {audio.shape}, Frecuencia: {orig_freq}Hz")
    print(f"   ✓ Rango de valores: [{audio.min():.4f}, {audio.max():.4f}]")
    print(f"   ✓ Dtype: {audio.dtype}")

    # Remuestrear
    if orig_freq != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq, 16000)
        audio = resampler(audio)
        print(f"   ✓ Audio remuestreado a 16kHz - Shape: {audio.shape}")

    # Convertir a mono
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
        print(f"   ✓ Audio convertido a mono - Shape: {audio.shape}")

    # Procesar un pequeño fragmento de PRUEBA (10 segundos)
    print("\n🧪 PRUEBA: Procesando 10 segundos de audio...")
    fragmento_prueba = audio[:, :160000]  # 10 segundos a 16kHz
    print(f"   ✓ Fragmento de prueba - Shape: {fragmento_prueba.shape}")
    print(f"   ✓ Rango: [{fragmento_prueba.min():.4f}, {fragmento_prueba.max():.4f}]")

    # Preparar entrada
    print("\n🔧 Preparando entrada del procesador...")
    audio_inputs = processor(
        audio=fragmento_prueba.squeeze(0).numpy(),
        sampling_rate=16000,
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
            tgt_lang="spa",
            no_repeat_ngram_size=5,
            repetition_penalty=1.3,
            num_beams=2,
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

    torchaudio.save("audio_prueba_traducido.wav", audio_tensor_cpu, 16000)
    print(f"   ✓ Guardado: audio_prueba_traducido.wav")

    print("\n✅ Debug completado. Revisa el archivo 'audio_prueba_traducido.wav' para escuchar la calidad del audio traducido.")

if __name__ == "__main__":
    debug_audio_translation()
