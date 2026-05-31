"""Stem separator — isolates vocals from background using Demucs htdemucs model.

Uses 4-stem separation (drums, bass, other, vocals) and sums the non-vocal
stems into a single 'no_vocals' track that is preserved unchanged.
"""

import tempfile
from pathlib import Path

import torch
import torchaudio


class StemSeparator:
    """Separates vocals from background audio using Demucs (htdemucs model).

    The model (~80 MB) is downloaded once to ~/.cache/torch/hub/checkpoints/
    and loaded lazily on first call to separate().

    Usage:
        sep = StemSeparator()
        vocals_path, no_vocals_path, tmp_dir = sep.separate(video_path)
        # ... process vocals_path ...
        shutil.rmtree(tmp_dir)  # caller must clean up
    """

    MODEL_NAME = "htdemucs"

    def __init__(self) -> None:
        self._model = None
        self._samplerate: int = 44100

    # ── Public API ────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load htdemucs model (idempotent). Downloads ~80 MB on first run."""
        if self._model is not None:
            return
        from demucs.pretrained import get_model  # imported lazily

        print(f"🎵 Cargando Demucs ({self.MODEL_NAME})...")
        self._model = get_model(self.MODEL_NAME)
        self._samplerate = self._model.samplerate
        print(
            f"   ✓ Demucs listo — SR: {self._samplerate} Hz  |  "
            f"stems: {self._model.sources}"
        )

    def separate(self, audio_path: Path) -> tuple[Path, Path, Path]:
        """Separate audio/video into vocals and no_vocals WAV files.

        Args:
            audio_path: Path to any audio or video file supported by ffmpeg.

        Returns:
            (vocals_path, no_vocals_path, tmp_dir)
            vocals_path:    WAV with only the vocal track  (44100 Hz stereo)
            no_vocals_path: WAV with drums+bass+other      (44100 Hz stereo)
            tmp_dir:        Temp directory — caller is responsible for deleting it.
        """
        self.load_model()
        print(f"🎵 Separando stems: {audio_path.name}...")

        from demucs.audio import AudioFile
        from demucs.apply import apply_model

        # Load audio at model's sample rate (44100 Hz) in stereo
        wav = AudioFile(audio_path).read(
            streams=0,
            samplerate=self._samplerate,
            channels=2,
        )

        # Normalize (same as demucs reference implementation)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()

        # Run model — output shape: [stems, channels, samples]
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"   ⚙️  Ejecutando htdemucs en {device}...")
        with torch.no_grad():
            sources = apply_model(
                self._model,
                wav[None],
                device=device,
                progress=True,
                overlap=0.25,
            )[0]

        # Denormalize
        sources = sources * ref.std() + ref.mean()

        stem_names  = self._model.sources              # ['drums', 'bass', 'other', 'vocals']
        vocals_idx  = stem_names.index("vocals")
        vocals      = sources[vocals_idx].cpu()        # [channels, samples]
        no_vocals   = sum(
            sources[i].cpu()
            for i in range(len(stem_names))
            if i != vocals_idx
        )

        # Save to temp directory
        tmp_dir        = Path(tempfile.mkdtemp(prefix="m4t_stems_"))
        vocals_path    = tmp_dir / "vocals.wav"
        no_vocals_path = tmp_dir / "no_vocals.wav"

        torchaudio.save(str(vocals_path),    vocals,    self._samplerate)
        torchaudio.save(str(no_vocals_path), no_vocals, self._samplerate)

        dur = vocals.shape[1] / self._samplerate
        print(f"   ✓ vocals    → {vocals_path.name}  ({dur:.1f}s)")
        print(f"   ✓ no_vocals → {no_vocals_path.name}")
        return vocals_path, no_vocals_path, tmp_dir
