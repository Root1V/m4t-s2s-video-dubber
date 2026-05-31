import os
import glob
from datetime import datetime
from moviepy import VideoFileClip, AudioFileClip

# --- CONFIGURACIÓN ---
# Carpeta donde están los videos originales
CARPETA_ENTRADA = "/Users/emericespiritusantiago/Documents/Projects/video_translate/videos"

# Nombre del video dentro de CARPETA_ENTRADA
ARCHIVO_VIDEO = "video.mp4"

# Carpeta donde están los audios traducidos (misma que usa traducir.py)
CARPETA_SALIDA = "/Users/emericespiritusantiago/Documents/Projects/video_translate/resultados"

# Deja en None para tomar el WAV más reciente de CARPETA_SALIDA (recomendado),
# o especifica el nombre exacto relativo a CARPETA_SALIDA, ej: "video_esp_20260524_153000.wav"
ARCHIVO_AUDIO = None
# ---------------------


def seleccionar_audio(audio_especificado):
    """Selecciona el archivo de audio a usar. Si no se especifica, toma el más reciente de CARPETA_SALIDA."""
    if audio_especificado:
        ruta = os.path.join(CARPETA_SALIDA, audio_especificado) if not os.path.isabs(audio_especificado) else audio_especificado
        if not os.path.exists(ruta):
            print(f"❌ Error: No se encontró '{ruta}'.")
            return None
        return ruta

    # Buscar todos los WAVs generados por traducir.py en CARPETA_SALIDA
    patron = os.path.join(CARPETA_SALIDA, "*_esp_*.wav")
    wavs = sorted(glob.glob(patron), key=os.path.getmtime, reverse=True)
    if not wavs:
        print(f"❌ No se encontró ningún WAV traducido en '{CARPETA_SALIDA}'.")
        print("   Primero ejecuta 'python traducir.py' para generar la traducción.")
        return None

    if len(wavs) > 1:
        print(f"📂 Se encontraron {len(wavs)} archivos de audio traducido en '{CARPETA_SALIDA}':")
        for i, w in enumerate(wavs):
            size_mb = os.path.getsize(w) / (1024 * 1024)
            fecha = datetime.fromtimestamp(os.path.getmtime(w)).strftime("%Y-%m-%d %H:%M:%S")
            print(f"   [{i}] {os.path.basename(w)}  ({size_mb:.1f} MB, {fecha})")
        print(f"\n   ✓ Usando el más reciente: {os.path.basename(wavs[0])}")
        print(f"   (Cambia ARCHIVO_AUDIO en el script si quieres usar otro)")

    return wavs[0]


def main():
    ruta_video = os.path.join(CARPETA_ENTRADA, ARCHIVO_VIDEO)
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    if not os.path.exists(ruta_video):
        print(f"❌ Error: No se encontró '{ruta_video}'.")
        return

    archivo_audio = seleccionar_audio(ARCHIVO_AUDIO)
    if archivo_audio is None:
        return

    # Nombre del video de salida: <nombre_original>_esp_<timestamp>.mp4
    nombre_base = os.path.splitext(ARCHIVO_VIDEO)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo_salida = os.path.join(CARPETA_SALIDA, f"{nombre_base}_esp_{timestamp}.mp4")

    print(f"\n🎬 Ensamblando video:")
    print(f"   Video original : {ruta_video}")
    print(f"   Audio traducido: {archivo_audio}")
    print(f"   Salida         : {archivo_salida}")

    try:
        video = VideoFileClip(ruta_video)
        audio = AudioFileClip(archivo_audio)

        duracion_video = video.duration
        duracion_audio = audio.duration
        print(f"\n⏱️  Duración video: {duracion_video:.2f}s | Duración audio: {duracion_audio:.2f}s")

        if abs(duracion_audio - duracion_video) > 1.0:
            print(f"⚠️  Diferencia de duración > 1s. Ajustando audio al largo del video...")
            audio = audio.with_duration(duracion_video)

        video_final = video.with_audio(audio)

        print("\n⏳ Codificando video... (puede tardar varios minutos)")
        video_final.write_videofile(
            archivo_salida,
            codec="libx264",
            audio_codec="aac",
        )

        video.close()
        audio.close()
        video_final.close()

        print(f"\n🎉 ¡Listo! Video guardado en: '{archivo_salida}'")

    except Exception as e:
        print(f"❌ Error al procesar el video: {e}")


if __name__ == "__main__":
    main()

