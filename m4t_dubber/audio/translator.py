"""Audio translator — speech-to-speech using SeamlessM4T v2.

Pipeline:
  1. Load & resample audio to 16 kHz mono
  2. VAD segmentation (RMS energy)
  3. Translate each speech segment (sub-chunked to stay within MPS limits)
  4. Stretch translated chunks back to original duration
  5. Concatenate silence + speech → normalize → save WAV
"""

import math
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToSpeech

from m4t_dubber import config


class AudioTranslator:
    """Translates the audio track of a video from English to Spanish."""

    def __init__(self) -> None:
        self._device: torch.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self._processor: AutoProcessor | None = None
        self._model: SeamlessM4Tv2ForSpeechToSpeech | None = None

    # ── Public API ────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load model and processor (idempotent)."""
        if self._model is not None:
            return
        print(f"🤖 Cargando {config.MODEL_ID} en {self._device}...")
        self._processor = AutoProcessor.from_pretrained(config.MODEL_ID)
        self._model = (
            SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(config.MODEL_ID).to(self._device)
        )

    def translate(
        self,
        input_path: Path,
        output_path: Path,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
    ) -> tuple[Path, list[tuple[float, float, str]]]:
        """Translate audio track and save as WAV.

        Returns:
            (output_path, subtitle_entries) where subtitle_entries is a list of
            (start_seconds, end_seconds, translated_text) tuples, one per speech segment.

        Args:
            src_lang: Source language code (e.g. "eng"). Defaults to config.SRC_LANG.
            tgt_lang: Target language code (e.g. "spa", "fra", "por"). Defaults to config.TGT_LANG.
        """
        src_lang = src_lang or config.SRC_LANG
        tgt_lang = tgt_lang or config.TGT_LANG

        self.load_model()
        print(f"📂 Entrada : {input_path}")
        print(f"📂 Salida  : {output_path}")

        audio = self._load_audio(input_path)
        original_samples = audio.shape[1]
        print(f"   ✓ Duración original: {original_samples / config.SAMPLE_RATE:.2f}s")

        print("\n🔍 Detectando segmentos de voz y silencio...")
        segments = _find_speech_segments(audio.squeeze(0))
        n_speech  = sum(1 for s in segments if s[2])
        n_silence = sum(1 for s in segments if not s[2])
        speech_s  = sum(s[1] - s[0] for s in segments if s[2]) / config.SAMPLE_RATE
        print(f"   ✓ {n_speech} segmentos de voz ({speech_s:.1f}s), {n_silence} de silencio")

        print(f"\n🗣️ Traduciendo {src_lang} → {tgt_lang} (speaker_id={config.SPEAKER_ID})...")
        chunks, subtitles = self._translate_segments(audio, segments, src_lang=src_lang, tgt_lang=tgt_lang)

        print("\n🎛️ Uniendo fragmentos...")
        audio_final = torch.cat(chunks, dim=1)
        audio_final = _ensure_2d(audio_final)
        audio_final = _normalize(audio_final)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), audio_final, config.SAMPLE_RATE)
        print(f"\n🎉 Audio guardado: '{output_path}'")
        return output_path, subtitles

    # ── Private helpers ───────────────────────────────────────────

    def _load_audio(self, path: Path) -> torch.Tensor:
        audio, sr = torchaudio.load(str(path))
        if sr != config.SAMPLE_RATE:
            audio = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)(audio)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        return audio

    def _translate_segments(
        self,
        audio: torch.Tensor,
        segments: list[tuple[int, int, bool]],
        src_lang: str,
        tgt_lang: str,
    ) -> tuple[list[torch.Tensor], list[tuple[float, float, str]]]:
        seg_max   = config.SAMPLE_RATE * config.MAX_CHUNK_S
        seg_min   = int(config.SAMPLE_RATE * config.MIN_CHUNK_S)
        total     = audio.shape[1]
        result:    list[torch.Tensor] = []
        subtitles: list[tuple[float, float, str]] = []

        for idx, (start, end, is_speech) in enumerate(segments):
            seg_len = end - start

            if not is_speech:
                result.append(torch.zeros(1, seg_len))
                continue

            fragment   = audio[:, start:end]
            sub_out:   list[torch.Tensor] = []
            sub_texts: list[str] = []

            for j in range(0, seg_len, seg_max):
                chunk = fragment[:, j:j + seg_max]
                if chunk.shape[1] < seg_min:
                    sub_out.append(torch.zeros(1, chunk.shape[1]))
                    continue

                inputs = self._processor(
                    audio=chunk.squeeze(0).numpy(),
                    sampling_rate=config.SAMPLE_RATE,
                    return_tensors="pt",
                ).to(self._device)

                with torch.no_grad():
                    output = self._model.generate(
                        **inputs,
                        tgt_lang=tgt_lang,
                        speaker_id=config.SPEAKER_ID,
                        no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
                        repetition_penalty=config.REPETITION_PENALTY,
                        num_beams=config.NUM_BEAMS,
                        return_intermediate_token_ids=True,
                    )

                chunk_audio = output.waveform.cpu()
                sub_out.append(chunk_audio)

                if output.sequences is not None and len(output.sequences) > 0:
                    text = self._processor.decode(
                        output.sequences[0], skip_special_tokens=True
                    ).strip()
                    if text:
                        sub_texts.append(text)

                if self._device.type == "mps":
                    torch.mps.empty_cache()

            if not sub_out:
                result.append(torch.zeros(1, seg_len))
                continue

            seg_audio = torch.cat(sub_out, dim=1) if len(sub_out) > 1 else sub_out[0]
            seg_audio = _stretch(seg_audio, seg_len)
            result.append(seg_audio)

            if sub_texts:
                start_s = start / config.SAMPLE_RATE
                end_s   = end   / config.SAMPLE_RATE
                subtitles.append((start_s, end_s, " ".join(sub_texts)))

            pct = round(end / total * 100, 1)
            print(f"   Segmento {idx + 1}/{len(segments)} — {pct}%")

        return result, subtitles


# ── Module-level DSP helpers ──────────────────────────────────────


def _find_speech_segments(
    audio_1d: torch.Tensor,
) -> list[tuple[int, int, bool]]:
    """VAD by adaptive RMS threshold.

    Returns list of (start_sample, end_sample, is_speech).
    """
    frame_s = config.SAMPLE_RATE * config.VAD_FRAME_MS // 1000
    total   = audio_1d.shape[0]
    n_frames = total // frame_s

    if n_frames == 0:
        return [(0, total, False)]

    frames    = audio_1d[:n_frames * frame_s].reshape(n_frames, frame_s)
    rms       = torch.sqrt(torch.mean(frames ** 2, dim=1)).numpy()
    threshold = np.percentile(rms, config.VAD_PERCENTILE) * config.VAD_MULTIPLIER
    is_speech = rms > threshold

    # Fill short silent gaps to avoid cutting mid-utterance
    min_sil = max(1, config.VAD_MIN_SILENCE_MS // config.VAD_FRAME_MS)
    i = 0
    while i < len(is_speech):
        if not is_speech[i]:
            j = i
            while j < len(is_speech) and not is_speech[j]:
                j += 1
            if (j - i) < min_sil and i > 0 and j < len(is_speech):
                is_speech[i:j] = True
            i = j
        else:
            i += 1

    segments: list[tuple[int, int, bool]] = []
    cur_type, cur_start = bool(is_speech[0]), 0
    for f in range(1, n_frames):
        if bool(is_speech[f]) != cur_type:
            segments.append((cur_start * frame_s, f * frame_s, cur_type))
            cur_type, cur_start = bool(is_speech[f]), f
    segments.append((cur_start * frame_s, total, cur_type))
    return segments


def _stretch(audio: torch.Tensor, target: int) -> torch.Tensor:
    """Time-stretch to target sample count (phase vocoder, no pitch shift)."""
    current = audio.shape[-1]
    if current == 0 or target == 0:
        return torch.zeros(1, target)
    if current == target:
        return audio

    rate = current / target
    if rate < 0.33:  # > 3x stretch → pad with silence instead
        return torch.cat([audio, torch.zeros(1, target - current)], dim=-1)

    n_fft, hop = 512, 128
    window = torch.hann_window(n_fft)
    stft   = torch.stft(
        audio.squeeze(0), n_fft=n_fft, hop_length=hop,
        window=window, return_complex=True,
    )
    phase  = torch.linspace(0, math.pi * hop, n_fft // 2 + 1).unsqueeze(-1)
    stft_s = torchaudio.functional.phase_vocoder(stft, rate, phase)
    out    = torch.istft(stft_s, n_fft=n_fft, hop_length=hop, window=window, length=target)
    return out.unsqueeze(0)


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 3:
        t = t.squeeze(0)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t


def _normalize(t: torch.Tensor) -> torch.Tensor:
    peak = t.abs().max()
    return t / peak if peak > 0 else t
