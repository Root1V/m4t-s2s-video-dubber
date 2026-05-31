"""Dubbing pipeline — orchestrates translation + assembly for one or many videos."""

import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torchaudio

from m4t_dubber import config
from m4t_dubber.audio.assembler import VideoAssembler
from m4t_dubber.audio.separator import StemSeparator
from m4t_dubber.audio.subtitler import write_srt
from m4t_dubber.audio.translator import AudioTranslator
from m4t_dubber.audio.voice_cloner import VoiceCloner


class DubbingPipeline:
    """End-to-end pipeline: video(s) in → dubbed MP4(s) out.

    Usage:
        pipeline = DubbingPipeline()
        pipeline.run()                  # all videos in INPUT_DIR
        pipeline.run("tutorial.mp4")    # specific video
    """

    def __init__(self) -> None:
        self.translator = AudioTranslator()
        self.assembler  = VideoAssembler()
        self.separator  = StemSeparator()
        self.cloner     = VoiceCloner()

    # ── Public API ────────────────────────────────────────────────

    def run(
        self,
        video_name: str | None = None,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
        generate_srt: bool = False,
        use_stem: bool = False,
        clone_voice: bool = False,
    ) -> None:
        """Process one or all videos and print a summary.

        Args:
            src_lang:     Source language code (e.g. "eng"). Defaults to M4T_SRC_LANG env var.
            tgt_lang:     Target language code (e.g. "spa", "fra", "por"). Defaults to M4T_TGT_LANG env var.
            generate_srt: If True, write a .srt subtitle file alongside the output. Default: False.
            use_stem:     If True, separate vocals from background before translating. Default: False.
            clone_voice:  If True, synthesize translated speech in the original speaker's voice
                          using F5-TTS zero-shot voice cloning. Default: False.
        """
        videos = self._collect_videos(video_name)

        _banner(f"PIPELINE — {len(videos)} video(s) en cola")
        for i, v in enumerate(videos, 1):
            print(f"  {i}. {v.name}")

        exitosos: list[tuple[str, str]] = []
        fallidos:  list[tuple[str, str]] = []
        t_inicio = datetime.now()

        for i, video_path in enumerate(videos, 1):
            print(f"\n{'─' * 64}\n  [{i}/{len(videos)}] {video_path.name}\n{'─' * 64}")
            t0 = datetime.now()
            try:
                self._process(video_path, src_lang=src_lang, tgt_lang=tgt_lang, generate_srt=generate_srt, use_stem=use_stem, clone_voice=clone_voice)
                self._move_to_processed(video_path)
                duracion = str(datetime.now() - t0).split(".")[0]
                exitosos.append((video_path.name, duracion))
                print(f"\n✅  {video_path.name}  — {duracion}")
            except Exception as e:
                fallidos.append((video_path.name, str(e)))
                print(f"\n❌  Error en '{video_path.name}':")
                traceback.print_exc()

        _banner("RESUMEN FINAL")
        print(f"  Tiempo total: {str(datetime.now() - t_inicio).split('.')[0]}")
        print(f"\n  ✅ Exitosos ({len(exitosos)}):")
        for nombre, t in exitosos:
            print(f"     • {nombre}  ({t})")
        if fallidos:
            print(f"\n  ❌ Fallidos ({len(fallidos)}):")
            for nombre, err in fallidos:
                print(f"     • {nombre}: {err}")
        print()

    # ── Private helpers ───────────────────────────────────────────

    def _process(
        self,
        video_path: Path,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
        generate_srt: bool = False,
        use_stem: bool = False,
        clone_voice: bool = False,
    ) -> None:
        resolved_tgt = tgt_lang or config.TGT_LANG
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem     = video_path.stem
        wav_path = config.OUTPUT_DIR / f"{stem}_{resolved_tgt}_{ts}.wav"
        mp4_path = config.OUTPUT_DIR / f"{stem}_{resolved_tgt}_{ts}.mp4"
        srt_path = config.OUTPUT_DIR / f"{stem}_{resolved_tgt}_{ts}.srt"

        if use_stem and clone_voice:
            # ── Stem separation + voice cloning ───────────────────
            _banner(f"[SEPARANDO STEMS] {video_path.name}")
            vocals_path, no_vocals_path, tmp_dir = self.separator.separate(video_path)
            try:
                _banner(f"[TRADUCIENDO (texto)] {video_path.name}")
                dummy_wav = tmp_dir / "vocals_translated.wav"
                _, subtitles = self.translator.translate(
                    vocals_path, dummy_wav, src_lang=src_lang, tgt_lang=tgt_lang
                )
                ref_path = tmp_dir / "reference.wav"
                self.cloner.extract_reference(vocals_path, ref_path)
                _wav, _sr = torchaudio.load(str(dummy_wav))
                total_dur = _wav.shape[1] / _sr
                del _wav
                vocals_cloned = tmp_dir / "vocals_cloned.wav"
                _banner(f"[CLONANDO VOZ] {video_path.name}")
                self.cloner.synthesize_from_segments(
                    subtitles, ref_path, vocals_cloned, total_duration=total_dur
                )
                _banner(f"[MEZCLANDO STEMS] {video_path.name}")
                _mix_stems(vocals_cloned, no_vocals_path, wav_path)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        elif clone_voice:
            # ── Voice cloning (standard audio, no stem separation) ─
            import tempfile as _tempfile

            with _tempfile.TemporaryDirectory(prefix="m4t_clone_") as tmpdir:
                tmp_path  = Path(tmpdir)
                dummy_wav = tmp_path / "translated_dummy.wav"
                ref_path  = tmp_path / "reference.wav"

                _banner(f"[TRADUCIENDO (texto)] {video_path.name}")
                _, subtitles = self.translator.translate(
                    video_path, dummy_wav, src_lang=src_lang, tgt_lang=tgt_lang
                )
                self.cloner.extract_reference(video_path, ref_path)
                _wav, _sr = torchaudio.load(str(dummy_wav))
                total_dur = _wav.shape[1] / _sr
                del _wav

                _banner(f"[CLONANDO VOZ] {video_path.name}")
                self.cloner.synthesize_from_segments(
                    subtitles, ref_path, wav_path, total_duration=total_dur
                )

        elif use_stem:
            # ── Stem-separated pipeline ───────────────────────────
            _banner(f"[SEPARANDO STEMS] {video_path.name}")
            vocals_path, no_vocals_path, tmp_dir = self.separator.separate(video_path)
            try:
                _banner(f"[TRADUCIENDO VOCES] {video_path.name}")
                vocals_translated = tmp_dir / "vocals_translated.wav"
                _, subtitles = self.translator.translate(
                    vocals_path, vocals_translated, src_lang=src_lang, tgt_lang=tgt_lang
                )
                _banner(f"[MEZCLANDO STEMS] {video_path.name}")
                _mix_stems(vocals_translated, no_vocals_path, wav_path)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            # ── Standard pipeline ─────────────────────────────────
            _banner(f"[TRADUCIENDO] {video_path.name}")
            _, subtitles = self.translator.translate(
                video_path, wav_path, src_lang=src_lang, tgt_lang=tgt_lang
            )

        if generate_srt and subtitles:
            write_srt(subtitles, srt_path)
            print(f"📝 Subtítulos: '{srt_path}'")

        _banner(f"[ENSAMBLANDO] {video_path.name} → {resolved_tgt}")
        self.assembler.assemble(video_path, wav_path, mp4_path)

    def _collect_videos(self, name: str | None) -> list[Path]:
        if name:
            p = Path(name) if Path(name).is_absolute() else config.INPUT_DIR / name
            if not p.exists():
                print(f"❌ No se encontró: {p}")
                sys.exit(1)
            return [p]

        videos = sorted(
            p
            for ext in config.VIDEO_EXTENSIONS
            for p in config.INPUT_DIR.glob(f"*{ext}")
        )
        if not videos:
            print(f"❌ No hay videos en: {config.INPUT_DIR}")
            print(f"   Extensiones soportadas: {', '.join(config.VIDEO_EXTENSIONS)}")
            sys.exit(1)
        return videos

    def _move_to_processed(self, src: Path) -> None:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        dst = config.PROCESSED_DIR / src.name
        if dst.exists():
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            dst = config.PROCESSED_DIR / f"{src.stem}_{ts}{src.suffix}"
        shutil.move(str(src), str(dst))
        print(f"📦 Movido a procesados: {dst.name}")


# ── Helpers ───────────────────────────────────────────────────────


def _mix_stems(vocals_path: Path, no_vocals_path: Path, output_path: Path) -> None:
    """Mix translated vocals with preserved background stems into a single WAV.

    Handles sample-rate mismatch (vocals are at 16 kHz, background at 44.1 kHz)
    and length differences caused by the translation phase-vocoder stretch.
    """
    v,  sr_v  = torchaudio.load(str(vocals_path))
    bg, sr_bg = torchaudio.load(str(no_vocals_path))

    # Resample translated vocals to match background sample rate
    if sr_v != sr_bg:
        v = torchaudio.transforms.Resample(sr_v, sr_bg)(v)

    # Broadcast mono vocals to stereo if background is stereo
    if bg.shape[0] > v.shape[0]:
        v = v.repeat(bg.shape[0], 1)

    # Align lengths (translation may shift total duration slightly)
    min_len = min(v.shape[1], bg.shape[1])
    v  = v[:,  :min_len]
    bg = bg[:, :min_len]

    # Mix and normalize to avoid clipping (leave 5% headroom)
    mixed = v + bg
    peak  = mixed.abs().max()
    if peak > 1e-7:
        mixed = mixed / peak * 0.95

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), mixed, sr_bg)
    print(f"🎚️  Mezcla guardada: '{output_path.name}'")


def _banner(title: str, width: int = 64) -> None:
    print(f"\n{'═' * width}\n  {title}\n{'═' * width}")
