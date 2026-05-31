"""
Script para probar diferentes speaker_id del vocoder y encontrar la voz masculina.
Genera un archivo WAV por cada ID con los primeros 10s del video.
Escucha los archivos y apunta el ID que suene más masculino en español.
Luego cambia SPEAKER_GENDER = "male" y agrega el ID encontrado en traducir.py
"""
import torch
import torchaudio
import os
from transformers import SeamlessM4Tv2ForSpeechToSpeech, AutoProcessor

# --- CONFIGURACIÓN ---
ARCHIVO_ENTRADA = "video.mp4"
IDS_A_PROBAR = list(range(10))   # Prueba IDs 0-9; cambia si quieres más
SEGUNDOS = 10                     # Segundos del video a usar para la prueba
# ---------------------

def main():
    if not os.path.exists(ARCHIVO_ENTRADA):
        print(f"❌ No se encontró '{ARCHIVO_ENTRADA}'.")
        return

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🍏 Procesando en: {device}")

    print("🤖 Cargando modelo...")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
        "facebook/seamless-m4t-v2-large"
    ).to(device)

    print(f"📦 Cargando audio de: {ARCHIVO_ENTRADA}")
    audio, orig_freq = torchaudio.load(ARCHIVO_ENTRADA)

    if orig_freq != 16000:
        audio = torchaudio.transforms.Resample(orig_freq, 16000)(audio)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    muestra = audio[:, : 16000 * SEGUNDOS]

    audio_inputs = processor(
        audio=muestra.squeeze(0).numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    ).to(device)

    print(f"\n🎙️  Generando {len(IDS_A_PROBAR)} voces distintas ({SEGUNDOS}s cada una)...\n")

    for sid in IDS_A_PROBAR:
        with torch.no_grad():
            output = model.generate(
                **audio_inputs,
                tgt_lang="spa",
                speaker_id=sid,
                no_repeat_ngram_size=5,
                repetition_penalty=1.3,
            )
        audio_out = output[0] if isinstance(output, tuple) else output
        audio_out = audio_out.cpu()

        if len(audio_out.shape) == 1:
            audio_out = audio_out.unsqueeze(0)

        nombre = f"voz_id_{sid:03d}.wav"
        torchaudio.save(nombre, audio_out, 16000)
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
