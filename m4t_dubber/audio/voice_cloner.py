"""Voice cloner — zero-shot synthesis of translated speech in the original speaker's voice.

Uses F5-TTS (v1 Base) with a short reference audio clip (~15 s) extracted from the
original speaker.  The F5-TTS model (~1.5 GB) is downloaded on first use from
HuggingFace and cached at ~/.cache/huggingface/.

Cross-lingual cloning: reference audio can be in any language; the generated speech
targets the language of ``gen_text``.

Supported languages: en, es, fr, de, pt, it, ja, zh, ko, ru, ar, nl, pl, tr, hi …
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torchaudio


class VoiceCloner:
    """Zero-shot voice cloning with F5-TTS."""

    MODEL_NAME   = "F5TTS_v1_Base"
    SAMPLE_RATE  = 24_000   # F5-TTS native output sample rate
    REF_DURATION = 15.0     # seconds of reference audio used for voice conditioning

    def __init__(self) -> None:
        self._model = None

    # ── Public API ────────────────────────────────────────────────

    def load_model(self) -> None:
        """Lazy-load F5-TTS (~1.5 GB download on first run)."""
        if self._model is not None:
            return
        from f5_tts.api import F5TTS  # noqa: PLC0415

        print("🎙️ Cargando F5-TTS (primera ejecución: ~1.5 GB descarga)...")
        self._model = F5TTS(model=self.MODEL_NAME)
        print("   ✓ F5-TTS listo")

    def extract_reference(
        self,
        audio_path: Path,
        output_path: Path | None = None,
        duration_s: float = REF_DURATION,
    ) -> Path:
        """Save first ``duration_s`` seconds of ``audio_path`` as a speaker reference WAV.

        Args:
            audio_path:  Source audio or video file (any format torchaudio can read).
            output_path: Destination WAV.  A temp file is used when *None*.
            duration_s:  Maximum clip length in seconds (default 15 s).

        Returns:
            Path to the reference WAV.
        """
        wav, sr = torchaudio.load(str(audio_path))
        max_samples = int(sr * duration_s)
        wav = wav[:, :max_samples]

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, prefix="m4t_ref_"
            )
            output_path = Path(tmp.name)
            tmp.close()

        torchaudio.save(str(output_path), wav, sr)
        print(f"   ✓ Referencia de voz: {output_path.name}  ({wav.shape[1] / sr:.1f} s)")
        return output_path

    def synthesize_from_segments(
        self,
        segments: list[tuple[float, float, str]],
        reference_audio: Path,
        output_path: Path,
        total_duration: float | None = None,
    ) -> Path:
        """Synthesize each text segment in the cloned voice, preserving original timing.

        Each segment is placed at its original start timestamp with silence gaps between
        segments to maintain the original speech / pause structure.

        ``fix_duration`` is passed to F5-TTS so the synthesized clip matches the original
        segment window exactly — no post-hoc resampling needed.

        Args:
            segments:        ``(start_s, end_s, translated_text)`` from SeamlessM4T.
            reference_audio: Short WAV of the original speaker (3–60 s).
            output_path:     Destination WAV path (24 kHz, mono).
            total_duration:  Output length in seconds.  Defaults to end of last segment.

        Returns:
            ``output_path``
        """
        self.load_model()

        if not segments:
            raise ValueError("segments list is empty — nothing to synthesize")

        sr = self.SAMPLE_RATE
        if total_duration is None:
            total_duration = segments[-1][1]

        total_samples = int(total_duration * sr)
        output = torch.zeros(1, total_samples)

        print(f"🎙️ Sintetizando {len(segments)} segmento(s) con voz clonada...")
        print(f"   Referencia: {reference_audio.name}")

        for i, (start_s, end_s, text) in enumerate(segments, 1):
            if not text.strip():
                continue

            seg_duration  = end_s - start_s
            start_sample  = int(start_s * sr)
            short_text    = text[:60] + ("…" if len(text) > 60 else "")
            print(f"   [{i}/{len(segments)}] {start_s:.1f}s–{end_s:.1f}s: {short_text}")

            try:
                wav_np, seg_sr, _ = self._model.infer(
                    ref_file=str(reference_audio),
                    ref_text="",            # auto-transcribe reference via faster-whisper
                    gen_text=text,
                    fix_duration=seg_duration,  # match original segment window
                    remove_silence=False,
                )

                seg_wav = torch.from_numpy(wav_np).float().unsqueeze(0)

                # Resample to output SR if model returned a different rate
                if seg_sr != sr:
                    seg_wav = torchaudio.transforms.Resample(seg_sr, sr)(seg_wav)

                # Clip or pad to the exact target window
                target = int(seg_duration * sr)
                if seg_wav.shape[1] < target:
                    pad = torch.zeros(1, target - seg_wav.shape[1])
                    seg_wav = torch.cat([seg_wav, pad], dim=1)
                else:
                    seg_wav = seg_wav[:, :target]

                # Write into output buffer
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
