"""Video assembler — replaces the audio track of a video with a translated WAV."""

from pathlib import Path

from moviepy import AudioFileClip, VideoFileClip


class VideoAssembler:
    """Combines an original video with a translated WAV to produce a dubbed MP4."""

    def assemble(self, video_path: Path, audio_path: Path, output_path: Path) -> Path:
        """Merge video + audio and write the output MP4. Returns output_path."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n🎬 Ensamblando video:")
        print(f"   Video : {video_path}")
        print(f"   Audio : {audio_path}")
        print(f"   Salida: {output_path}")

        video = VideoFileClip(str(video_path))
        audio = AudioFileClip(str(audio_path))

        print(f"\n⏱️  Video: {video.duration:.2f}s | Audio: {audio.duration:.2f}s")

        if abs(audio.duration - video.duration) > 1.0:
            print("⚠️  Diferencia > 1s. Ajustando audio al largo del video...")
            audio = audio.with_duration(video.duration)

        final = video.with_audio(audio)
        print("\n⏳ Codificando... (puede tardar varios minutos)")
        final.write_videofile(str(output_path), codec="libx264", audio_codec="aac")

        video.close()
        audio.close()
        final.close()

        print(f"\n🎉 Video guardado: '{output_path}'")
        return output_path

    @staticmethod
    def latest_wav(directory: Path) -> Path | None:
        """Return the most recently modified *_esp_*.wav in directory, or None."""
        wavs = sorted(
            directory.glob("*_esp_*.wav"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return wavs[0] if wavs else None
