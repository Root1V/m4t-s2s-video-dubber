"""Voice cloner — clonación de voz cross-lingüe del hablante original.

Usa XTTS v2 (Coqui TTS) para síntesis zero-shot cross-lingüe.  El modelo (~1.8 GB)
se descarga en el primer uso desde HuggingFace y se cachea en ~/.local/share/tts/.

Clonación cross-lingüe: la referencia puede estar en cualquier idioma (p.ej. inglés);
el audio generado estará en el idioma destino (p.ej. español) con fonética nativa
y el timbre de voz del hablante original — sin acento extranjero.

Idiomas soportados: es, en, fr, de, pt, it, ja, zh-cn, ko, ru, ar, nl, pl, tr, hi …
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio

# SeamlessM4T 3-letter codes → XTTS 2-letter codes
_LANG_MAP: dict[str, str] = {
    "spa": "es", "eng": "en", "fra": "fr", "deu": "de", "por": "pt",
    "ita": "it", "jpn": "ja", "cmn": "zh-cn", "ara": "ar", "rus": "ru",
    "nld": "nl", "pol": "pl", "tur": "tr", "kor": "ko", "hin": "hi",
    "hun": "hu", "ces": "cs",
    # también acepta códigos cortos directamente
    "es": "es", "en": "en", "fr": "fr", "de": "de", "pt": "pt",
    "it": "it", "ja": "ja", "ar": "ar", "ru": "ru", "nl": "nl",
    "pl": "pl", "tr": "tr", "ko": "ko", "hi": "hi",
}


class VoiceCloner:
    """Clonación de voz cross-lingüe con XTTS v2."""

    MODEL_NAME   = "tts_models/multilingual/multi-dataset/xtts_v2"
    SAMPLE_RATE  = 24_000   # tasa de muestreo nativa de XTTS v2
    REF_DURATION = 15.0     # segundos de referencia usados para condicionamiento

    def __init__(self) -> None:
        self._model = None

    # ── Public API ────────────────────────────────────────────────

    def load_model(self) -> None:
        """Lazy-load XTTS v2 (~1.8 GB descarga en primer uso)."""
        if self._model is not None:
            return
        from TTS.api import TTS  # noqa: PLC0415

        # Aceptar términos de uso de Coqui automáticamente
        os.environ.setdefault("COQUI_TOS_AGREED", "1")

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"🎙️ Cargando XTTS v2 (primera ejecución: ~1.8 GB descarga)...")
        self._model = TTS(self.MODEL_NAME)
        try:
            self._model.to(device)
            print(f"   ✓ XTTS v2 listo en {device}")
        except Exception:
            self._model.to("cpu")
            print("   ✓ XTTS v2 listo en cpu")

    def extract_reference(
        self,
        audio_path: Path,
        output_path: Path | None = None,
        duration_s: float = REF_DURATION,
        start_s: float = 0.0,
    ) -> Path:
        """Save ``duration_s`` seconds of ``audio_path`` starting at ``start_s`` as reference WAV.

        Args:
            audio_path:  Source audio or video file (any format torchaudio can read).
            output_path: Destination WAV.  A temp file is used when *None*.
            duration_s:  Maximum clip length in seconds (default 15 s).
            start_s:     Start offset in seconds.  Use to skip music intros and go straight
                         to the first actual speech segment (default 0).

        Returns:
            Path to the reference WAV.
        """
        wav, sr = torchaudio.load(str(audio_path))
        start_sample = int(sr * start_s)
        wav = wav[:, start_sample:]
        max_samples = int(sr * duration_s)
        wav = wav[:, :max_samples]

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, prefix="m4t_ref_"
            )
            output_path = Path(tmp.name)
            tmp.close()

        torchaudio.save(str(output_path), wav, sr)
        offset_info = f", desde {start_s:.1f}s" if start_s > 0 else ""
        print(f"   ✓ Referencia de voz: {output_path.name}  ({wav.shape[1] / sr:.1f} s{offset_info})")
        return output_path

    def synthesize_from_segments(
        self,
        segments: list[tuple[float, float, str]],
        reference_audio: Path,
        output_path: Path,
        total_duration: float | None = None,
        language: str = "es",
    ) -> Path:
        """Sintetiza cada segmento de texto con la voz clonada, preservando el timing original.

        Usa XTTS v2 para clonación cross-lingüe: el timbre del hablante original se
        preserva, pero la fonética es nativa del idioma destino (sin acento extranjero).

        Args:
            segments:        ``(start_s, end_s, texto_traducido)`` de SeamlessM4T.
            reference_audio: WAV corto del hablante original (3–60 s).
            output_path:     Ruta WAV de salida (24 kHz, mono).
            total_duration:  Duración total en segundos. Por defecto: fin del último segmento.
            language:        Código de idioma destino (3 letras SeamlessM4T o 2 letras XTTS).

        Returns:
            ``output_path``
        """
        self.load_model()

        if not segments:
            raise ValueError("segments list is empty — nothing to synthesize")

        # Resolver código de idioma al formato XTTS
        xtts_lang = _LANG_MAP.get(language, language[:2].lower())

        sr = self.SAMPLE_RATE
        if total_duration is None:
            total_duration = segments[-1][1]

        total_samples = int(total_duration * sr)
        output = torch.zeros(1, total_samples)

        print(f"🎙️ Sintetizando {len(segments)} segmento(s) con voz clonada ({xtts_lang})...")
        print(f"   Referencia: {reference_audio.name}")

        for i, (start_s, end_s, text) in enumerate(segments, 1):
            if not text.strip():
                continue

            # Permitir que el audio ocupe hasta el inicio del siguiente segmento
            next_start_s    = segments[i][0] if i < len(segments) else total_duration
            max_seg_samples = int((next_start_s - start_s) * sr)
            start_sample    = int(start_s * sr)
            short_text      = text[:60] + ("…" if len(text) > 60 else "")
            print(f"   [{i}/{len(segments)}] {start_s:.1f}s–{end_s:.1f}s: {short_text}")

            try:
                wav_data = self._model.tts(
                    text=text,
                    speaker_wav=str(reference_audio),
                    language=xtts_lang,
                )
                wav_np  = np.array(wav_data, dtype=np.float32)
                seg_wav = torch.from_numpy(wav_np).unsqueeze(0)

                # Resamplear si el modelo devuelve una tasa distinta
                model_sr = getattr(self._model.synthesizer, "output_sample_rate", sr)
                if model_sr != sr:
                    seg_wav = torchaudio.transforms.Resample(model_sr, sr)(seg_wav)

                # Recortar para no solapar con el siguiente segmento
                seg_wav = seg_wav[:, :max_seg_samples]

                # Escribir en el buffer de salida
                end_sample = min(start_sample + seg_wav.shape[1], total_samples)
                length     = end_sample - start_sample
                output[:, start_sample:end_sample] = seg_wav[:, :length]

            except Exception as exc:  # noqa: BLE001
                print(f"   ⚠️  Segmento {i} fallido — {exc}")

        # Normalize to −0.5 dBFS
        peak = output.abs().max()
        if peak > 1e-7:
            output = output / peak * 0.95

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), output, sr)
        print(f"\n🎉 Voz clonada guardada: '{output_path}'")
        return output_path
