import torch
import torchaudio
import os
import math
import numpy as np
from datetime import datetime
from transformers import SeamlessM4Tv2ForSpeechToSpeech, AutoProcessor

# --- CONFIGURACIÓN ---
# Carpeta donde están los videos a traducir
CARPETA_ENTRADA = "/Users/emericespiritusantiago/Documents/Projects/video_translate/videos"

# Nombre del video dentro de CARPETA_ENTRADA
ARCHIVO_ENTRADA = "video.mp4"

# Carpeta donde se guardarán los audios traducidos
CARPETA_SALIDA = "/Users/emericespiritusantiago/Documents/Projects/video_translate/resultados"

# Voz de síntesis. Opciones:
#   "auto"          → detección automática por tono (poco fiable con música de fondo)
#   "female"        → speaker_id=0 (femenino por defecto del modelo)
#   "male"          → speaker_id=1 (masculino aproximado; usa probar_voces.py para afinar)
#   <número 0-199>  → ID exacto encontrado con probar_voces.py  (ej: SPEAKER_GENDER = 4)
SPEAKER_GENDER = 4
# ---------------------

def detect_gender(audio_tensor, sample_rate=16000):
    """Detecta género del hablante por análisis de tono fundamental (F0)."""
    audio_np = audio_tensor.squeeze().numpy()
    segment = audio_np[:sample_rate * 10]  # Primeros 10 segundos

    min_period = int(sample_rate / 300)  # máx 300 Hz
    max_period = int(sample_rate / 80)   # mín 80 Hz
    frame_size = int(sample_rate * 0.04) # 40ms
    hop_size   = int(sample_rate * 0.01) # 10ms

    pitches = []
    for start in range(0, len(segment) - frame_size, hop_size):
        frame = segment[start:start + frame_size]
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]
        d = corr[min_period:max_period]
        if len(d) == 0:
            continue
        peak_idx = np.argmax(d) + min_period
        if corr[0] > 0 and d[peak_idx - min_period] > 0.3 * corr[0]:
            pitches.append(sample_rate / peak_idx)

    if not pitches:
        print("   🎤 No se detectó tono claro. Usando voz femenina por defecto.")
        return "female", 1

    median_pitch = np.median(pitches)
    print(f"   📊 Tono fundamental: {median_pitch:.1f} Hz")

    if median_pitch > 165:
        print("   🎤 Hablante detectada: Mujer (speaker_id=1)")
        return "female", 1
    else:
        print("   🎤 Hablante detectado: Hombre (speaker_id=0)")
        return "male", 0


def find_speech_segments(audio_1d, sr=16000, frame_ms=20, min_silence_ms=400, percentile=20, multiplier=3):
    """Detecta regiones de voz y silencio por energía RMS.
    Rellena silencios cortos (< min_silence_ms) para no cortar palabras.
    Retorna lista de (start_sample, end_sample, is_speech: bool).
    """
    frame_samples = int(sr * frame_ms / 1000)
    total_samples = audio_1d.shape[0]
    n_frames = total_samples // frame_samples

    if n_frames == 0:
        return [(0, total_samples, False)]

    frames = audio_1d[:n_frames * frame_samples].reshape(n_frames, frame_samples)
    rms = torch.sqrt(torch.mean(frames ** 2, dim=1)).numpy()

    # Umbral adaptativo: por encima del percentil bajo con margen
    threshold = np.percentile(rms, percentile) * multiplier
    is_speech = rms > threshold

    # Rellenar huecos de silencio cortos para evitar cortar mid-utterance
    min_silence_frames = max(1, int(min_silence_ms / frame_ms))
    i = 0
    while i < len(is_speech):
        if not is_speech[i]:
            j = i
            while j < len(is_speech) and not is_speech[j]:
                j += 1
            # Silencio muy corto dentro de voz → considerarlo voz
            if (j - i) < min_silence_frames and i > 0 and j < len(is_speech):
                is_speech[i:j] = True
            i = j
        else:
            i += 1

    # Construir lista de segmentos
    segments = []
    cur_type = bool(is_speech[0])
    cur_start = 0
    for f in range(1, n_frames):
        frame_type = bool(is_speech[f])
        if frame_type != cur_type:
            segments.append((cur_start * frame_samples, f * frame_samples, cur_type))
            cur_type = frame_type
            cur_start = f
    # Último segmento (incluye muestras restantes)
    segments.append((cur_start * frame_samples, total_samples, cur_type))

    return segments


def stretch_chunk(audio_tensor, target_samples):
    """Estira/comprime audio al número objetivo de muestras sin cambiar el tono.
    Si necesita estirar más de 3x, rellena con silencio (el modelo no generó suficiente)."""
    current_samples = audio_tensor.shape[-1]
    if current_samples == 0 or target_samples == 0:
        return torch.zeros(1, target_samples)
    if current_samples == target_samples:
        return audio_tensor

    rate = current_samples / target_samples  # < 1 = estirar, > 1 = comprimir

    if rate < 0.33:
        padding = torch.zeros(1, target_samples - current_samples)
        return torch.cat([audio_tensor, padding], dim=-1)

    n_fft = 512
    hop_length = 128
    window = torch.hann_window(n_fft)

    audio_1d = audio_tensor.squeeze(0)
    stft = torch.stft(audio_1d, n_fft=n_fft, hop_length=hop_length,
                      window=window, return_complex=True)
    phase_advance = torch.linspace(
        0, math.pi * hop_length, n_fft // 2 + 1
    ).unsqueeze(-1)
    stft_stretched = torchaudio.functional.phase_vocoder(stft, rate, phase_advance)
    audio_stretched = torch.istft(stft_stretched, n_fft=n_fft, hop_length=hop_length,
                                  window=window, length=target_samples)
    return audio_stretched.unsqueeze(0)


def main():
    # Resolver rutas de entrada y salida
    ruta_entrada = os.path.join(CARPETA_ENTRADA, ARCHIVO_ENTRADA)
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    if not os.path.exists(ruta_entrada):
        print(f"❌ Error: No se encontró '{ruta_entrada}'.")
        return

    # Nombre de salida: <nombre_original>_esp_<timestamp>.wav
    nombre_base = os.path.splitext(ARCHIVO_ENTRADA)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo_salida = os.path.join(CARPETA_SALIDA, f"{nombre_base}_esp_{timestamp}.wav")

    # 1. Configurar hardware
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🍏 Procesando en: {device}")
    print(f"📂 Entrada : {ruta_entrada}")
    print(f"📂 Salida  : {archivo_salida}")

    # 2. Cargar modelo
    print("🤖 Cargando SeamlessM4T v2 Large...")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained("facebook/seamless-m4t-v2-large").to(device)

    # 3. Cargar audio
    print(f"📦 Cargando audio de: {ruta_entrada}")
    audio, orig_freq = torchaudio.load(ruta_entrada)

    if orig_freq != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq, 16000)
        audio = resampler(audio)

    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    duracion_original = audio.shape[1]
    print(f"   ✓ Duración original: {duracion_original / 16000:.2f}s")

    # 4. Seleccionar voz de síntesis según género
    if SPEAKER_GENDER == "auto":
        print("\n🔍 Detectando género del hablante...")
        _, speaker_id = detect_gender(audio)
    elif isinstance(SPEAKER_GENDER, int):
        speaker_id = SPEAKER_GENDER
        print(f"\n🎤 Speaker ID configurado manualmente: {speaker_id}")
    elif SPEAKER_GENDER == "male":
        speaker_id = 1
        print(f"\n🎤 Género configurado: Hombre (speaker_id={speaker_id})")
        print("   💡 Usa probar_voces.py para encontrar el ID más masculino en español.")
    else:
        speaker_id = 0
        print(f"\n🎤 Género configurado: Mujer (speaker_id={speaker_id})")

    # 5. Detectar segmentos de voz y silencio (VAD)
    print("\n🔍 Detectando segmentos de voz y silencio...")
    audio_1d = audio.squeeze(0)
    segments = find_speech_segments(audio_1d)
    n_speech = sum(1 for s in segments if s[2])
    n_silence = sum(1 for s in segments if not s[2])
    total_speech_s = sum((s[1] - s[0]) for s in segments if s[2]) / 16000
    print(f"   ✓ {n_speech} segmentos de voz ({total_speech_s:.1f}s), {n_silence} segmentos de silencio")

    seg_max_samples = 16000 * 15  # máx 15s por sub-chunk (límite MPS)
    min_chunk_samples = 16000 * 1  # mínimo 1s para el modelo
    audios_traducidos_lista = []
    total_muestras = audio.shape[1]

    print(f"\n🗣️ Traduciendo Inglés → Español (speaker_id={speaker_id})...")

    for seg_idx, (seg_start, seg_end, is_speech_seg) in enumerate(segments):
        seg_samples = seg_end - seg_start

        if not is_speech_seg:
            # Silencio: preservar exactamente (no pasar por el modelo)
            audios_traducidos_lista.append(torch.zeros(1, seg_samples))
            continue

        # Segmento de voz: dividir en sub-chunks de máx 15s si es necesario
        fragmento = audio[:, seg_start:seg_end]
        sub_chunks_out = []

        for j in range(0, seg_samples, seg_max_samples):
            chunk_end = min(j + seg_max_samples, seg_samples)
            chunk = fragmento[:, j:chunk_end]

            if chunk.shape[1] < min_chunk_samples:
                # Demasiado corto para el modelo → silencio proporcional
                sub_chunks_out.append(torch.zeros(1, chunk.shape[1]))
                continue

            audio_inputs = processor(
                audio=chunk.squeeze(0).numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                output = model.generate(
                    **audio_inputs,
                    tgt_lang="spa",
                    speaker_id=speaker_id,
                    no_repeat_ngram_size=5,
                    repetition_penalty=1.3,
                    num_beams=2,
                )

            audio_chunk = output[0] if isinstance(output, tuple) else output
            sub_chunks_out.append(audio_chunk.cpu())

            if device.type == "mps":
                torch.mps.empty_cache()

        if not sub_chunks_out:
            audios_traducidos_lista.append(torch.zeros(1, seg_samples))
            continue

        # Concatenar sub-chunks del segmento
        seg_audio = torch.cat(sub_chunks_out, dim=1) if len(sub_chunks_out) > 1 else sub_chunks_out[0]

        # Estirar el segmento traducido al tamaño exacto del segmento original de voz
        seg_audio = stretch_chunk(seg_audio, seg_samples)
        audios_traducidos_lista.append(seg_audio)

        progreso = round((seg_end / total_muestras) * 100, 2)
        print(f"Segmento {seg_idx + 1}/{len(segments)} | Progreso: {progreso}%")

    # 6. Concatenar fragmentos (ya tienen la duración correcta por chunk)
    print("\n🎛️ Uniendo fragmentos...")
    audio_final = torch.cat(audios_traducidos_lista, dim=1)

    if len(audio_final.shape) == 3:
        audio_final = audio_final.squeeze(0)
    if len(audio_final.shape) == 1:
        audio_final = audio_final.unsqueeze(0)

    duracion_final = audio_final.shape[1]
    print(f"📊 Audio final: {duracion_final / 16000:.2f}s  |  Original: {duracion_original / 16000:.2f}s")

    # 7. Normalización global y guardar
    max_val = audio_final.abs().max()
    if max_val > 0:
        audio_final = audio_final / max_val

    torchaudio.save(archivo_salida, audio_final, 16000)
    print(f"\n🎉 ¡Listo! Archivo guardado: '{archivo_salida}'")

if __name__ == "__main__":
    main()

