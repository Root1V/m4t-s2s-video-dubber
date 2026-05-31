import torch
from transformers import SeamlessM4Tv2ForSpeechToSpeech, AutoProcessor
import torchaudio
import os
import numpy as np

# --- CONFIGURACIÓN DE ARCHIVOS ---
# Coloca tu video en la misma carpeta que este script con el nombre "video.mp4"
ARCHIVO_ENTRADA = "video.mp4"
ARCHIVO_AUDIO_TEMPORAL = "audio_extraido.wav"
ARCHIVO_SALIDA_ESPANOL = "tutorial_traducido_espanol.wav"
# ---------------------------------

def main():
    if not os.path.exists(ARCHIVO_ENTRADA):
        print(f"❌ Error: No se encontró el archivo '{ARCHIVO_ENTRADA}' en esta carpeta.")
        print("Por favor, copia tu video aquí y renómbralo a 'video.mp4'.")
        return

    # 1. Configurar el hardware del MacBook M4 Max (Memoria unificada Apple Silicon)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🍏 Procesando en el Chip de Apple mediante: {device}")

    # 2. Cargar el modelo grande v2 de Meta AI desde los servidores de Hugging Face
    print("🤖 Cargando SeamlessM4T v2 Large en memoria (Aprox. 10GB)...")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained("facebook/seamless-m4t-v2-large").to(device)

    # 3. Extraer, remuestrear y normalizar el canal de audio del video
    print(f"📦 Extrayendo y convirtiendo el audio de: {ARCHIVO_ENTRADA}")
    audio, orig_freq = torchaudio.load(ARCHIVO_ENTRADA)

    # Convertir la frecuencia a 16kHz (requisito obligatorio de la IA de Meta)
    if orig_freq != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq, 16000)
        audio = resampler(audio)

    # Convertir a canal Mono si el video viene codificado en Estéreo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # 4. Segmentación inteligente del audio
    # 15 segundos es el máximo seguro para MPS sin problemas de precisión numérica
    total_muestras = audio.shape[1]
    segmento_duracion = 16000 * 15  # 15 segundos a 16kHz
    audios_traducidos_lista = []

    print("\n🗣️ Comenzando traducción de Inglés a Español por bloques...")
    
    for i in range(0, total_muestras, segmento_duracion):
        fin = min(i + segmento_duracion, total_muestras)
        fragmento = audio[:, i:fin]
        
        # Omitir fragmentos muy pequeños (menores a 1 segundo)
        if fragmento.shape[1] < 16000:
            continue
            
        # Preparar tensores para los núcleos MPS del chip M4 Max
        audio_inputs = processor(audio=fragmento.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model.generate(
                **audio_inputs,
                tgt_lang="spa",
                no_repeat_ngram_size=5,       # Evita repetición de frases
                repetition_penalty=1.3,        # Penaliza tokens repetidos
                num_beams=2,                   # Mejor calidad de traducción
            )
        
        # Extraer tensor de audio de la tupla de salida
        if isinstance(output, tuple):
            audio_tensor = output[0]  # Primer elemento es el audio
        else:
            audio_tensor = output
        
        # Amplificar el audio (está muy bajo)
        # El modelo retorna audio muy silencioso (~±0.14), necesitamos amplificarlo
        # Normalización por fragmento — NO normalizar, dejar el rango natural
        # para evitar discontinuidades al concatenar
        
        audio_tensor = audio_tensor.cpu()
        audios_traducidos_lista.append(audio_tensor)
        
        # Liberar caché MPS entre bloques para evitar acumulación de errores numéricos
        if device.type == "mps":
            torch.mps.empty_cache()
        
        progreso = round((fin / total_muestras) * 100, 2)
        print(f"Progreso: {progreso}% completado...")

    # 5. Concatenar y exportar el resultado de la clase doblada
    print("\n🎛️ Uniendo todos los fragmentos procesados...")
    audio_final_espanol = torch.cat(audios_traducidos_lista, dim=1)

    # Ajustar dimensiones de salida para guardar el archivo (.wav)
    if len(audio_final_espanol.shape) == 3:
        audio_final_espanol = audio_final_espanol.squeeze(0)

    print(f"📊 Audio final - Shape: {audio_final_espanol.shape}, Dtype: {audio_final_espanol.dtype}")
    print(f"   Rango: [{audio_final_espanol.min():.6f}, {audio_final_espanol.max():.6f}]")

    # Normalización global única al final para preservar continuidad entre bloques
    max_val = audio_final_espanol.abs().max()
    if max_val > 0:
        audio_final_espanol = audio_final_espanol / max_val
        print(f"   ✓ Normalizado globalmente")

    # Asegurar formato correcto (1, samples)
    if len(audio_final_espanol.shape) == 1:
        audio_final_espanol = audio_final_espanol.unsqueeze(0)

    torchaudio.save(ARCHIVO_SALIDA_ESPANOL, audio_final_espanol, 16000)
    print(f"\n🎉 ¡Éxito absoluto! Tu tutorial traducido está listo en: '{ARCHIVO_SALIDA_ESPANOL}'")

if __name__ == "__main__":
    main()

