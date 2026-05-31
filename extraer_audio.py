import torchaudio
import os

# --- CONFIGURACIÓN ---
ARCHIVO_ENTRADA = "video.mp4"
ARCHIVO_SALIDA = "audio_original.wav"
# ----------------------

def main():
    if not os.path.exists(ARCHIVO_ENTRADA):
        print(f"❌ Error: No se encontró '{ARCHIVO_ENTRADA}'")
        return

    print(f"📦 Extrayendo audio de: {ARCHIVO_ENTRADA}")
    
    try:
        # Cargar audio directamente del video
        audio, orig_freq = torchaudio.load(ARCHIVO_ENTRADA)
        
        print(f"   ✓ Audio cargado - Shape: {audio.shape}, Frecuencia: {orig_freq}Hz")
        print(f"   ✓ Rango: [{audio.min():.4f}, {audio.max():.4f}]")
        
        # Guardar el audio original tal cual
        torchaudio.save(ARCHIVO_SALIDA, audio, orig_freq)
        
        print(f"\n✅ Audio extraído y guardado: '{ARCHIVO_SALIDA}'")
        print(f"   Duración: {audio.shape[1] / orig_freq:.2f} segundos")
        
    except Exception as e:
        print(f"❌ Error al extraer audio: {e}")

if __name__ == "__main__":
    main()
