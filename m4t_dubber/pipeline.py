"""Dubbing pipeline — orchestrates translation + assembly for one or many videos."""

import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

from m4t_dubber import config
from m4t_dubber.audio.assembler import VideoAssembler
from m4t_dubber.audio.translator import AudioTranslator


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

    # ── Public API ────────────────────────────────────────────────

    def run(self, video_name: str | None = None) -> None:
        """Process one or all videos and print a summary."""
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
                self._process(video_path)
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

    def _process(self, video_path: Path) -> None:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem     = video_path.stem
        wav_path = config.OUTPUT_DIR / f"{stem}_esp_{ts}.wav"
        mp4_path = config.OUTPUT_DIR / f"{stem}_esp_{ts}.mp4"

        _banner(f"[TRADUCIENDO] {video_path.name}")
        self.translator.translate(video_path, wav_path)

        _banner(f"[ENSAMBLANDO] {video_path.name}")
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


def _banner(title: str, width: int = 64) -> None:
    print(f"\n{'═' * width}\n  {title}\n{'═' * width}")
